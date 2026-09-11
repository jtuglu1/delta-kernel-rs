//! CRC (version checksum) file support.
//!
//! A [CRC file] contains a snapshot of table state at a specific version, which can be used to
//! optimize log replay operations like reading Protocol/Metadata, domain metadata, set
//! transactions, and ICT.
//!
//! [`Crc`] holds the in-memory state using shapes that make kernel queries easy: typed
//! state enums (`FileStatsState`, `DomainMetadataState`, `SetTransactionState`) and `HashMap`s
//! keyed by id, instead of the flat scalars and arrays of the on-disk format. It (de)serializes
//! to/from JSON via the private `CrcRaw` serde intermediate, which mirrors the wire format
//! exactly.
//!
//! [CRC file]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#version-checksum-file

// Allow unreachable_pub because this module is pub when internal-api is enabled
// but pub(crate) otherwise.
#![allow(unreachable_pub)]

mod delta;
mod file_size_histogram;
mod file_stats;
mod reader;
mod state;
mod writer;

use std::collections::HashMap;

#[allow(unused)]
pub(crate) use delta::{merge_domain_metadata, CrcDelta};
use delta_kernel_derive::internal_api;
pub use file_size_histogram::FileSizeHistogram;
pub use file_stats::FileStats;
#[allow(unused)]
pub(crate) use file_stats::{is_incremental_safe_operation, size_to_u64, FileStatsDelta};
pub(crate) use reader::read_crc_file_or_none;
#[cfg(test)]
pub(crate) use reader::try_read_crc_file;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
pub use state::{DomainMetadataState, FileStatsState, SetTransactionState};
#[allow(unused)]
pub(crate) use writer::try_write_crc_file;

use crate::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use crate::actions::{Add, DomainMetadata, Metadata, Protocol, SetTransaction};
use crate::table_features::TableFeature;
use crate::utils::require;
use crate::{DeltaResult, Error, Version};

// ============================================================================
// Crc: in-memory representation
// ============================================================================

/// Parsed content of a CRC (version checksum) file.
///
/// A `Crc` is either (a) loaded from disk (deserialized from a `.crc` JSON file via
/// the private `CrcRaw` intermediate) or (b) computed in memory (built incrementally via
/// `Crc::apply`).
///
/// A CRC file must:
/// 1. Be named `{version}.crc` with version zero-padded to 20 digits: `00000000000000000001.crc`
/// 2. Be stored directly in the _delta_log directory alongside Delta log files
/// 3. Contain exactly one JSON object with the schema mirrored by `CrcRaw`.
///
/// This struct and its fields are marked `pub`, but the `crc` module is only re-exported as `pub`
/// when the `internal-api` feature is enabled (otherwise `pub(crate)`). See `kernel/src/lib.rs`.
// TODO: rename `Crc` to `CrcState` to align with `FileStatsState`, `SetTransactionState`, etc.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Crc {
    // ===== Required fields =====
    /// The table version this CRC describes.
    pub version: Version,
    /// The table [`Metadata`] at this version.
    pub metadata: Metadata,
    /// The table [`Protocol`] at this version.
    pub protocol: Protocol,
    /// File-level statistics as a typed state. See [`FileStatsState`].
    pub(crate) file_stats_state: FileStatsState,

    // ===== Optional fields =====
    /// The in-commit timestamp of this version. Present iff In-Commit Timestamps are enabled.
    pub in_commit_timestamp_opt: Option<i64>,
    /// Active [`SetTransaction`] actions at this version, as a typed [`SetTransactionState`].
    /// `Complete(map)` is authoritative for misses; `Partial(map)` carries known-correct entries
    /// but requires log replay for misses. Only the `Complete` variant is persisted to the CRC
    /// file.
    pub set_transaction_state: SetTransactionState,
    /// Active (non-removed) [`DomainMetadata`] actions at this version, as a typed
    /// [`DomainMetadataState`]. Tombstones (`removed=true`) are never stored. `Complete(map)`
    /// is authoritative for misses; `Partial(map)` carries known-correct entries but requires
    /// log replay for misses. Only the `Complete` variant is persisted to the CRC file.
    ///
    /// TODO: when the table protocol does not enable the `domainMetadata` feature, no DM
    ///       action can exist, so `Partial(_)` is semantically equivalent to
    ///       `Complete(empty)` and both serde paths could collapse the distinction.
    pub domain_metadata_state: DomainMetadataState,

    // ===== Extended fields =====
    /// A unique identifier for the transaction that produced this commit.
    pub(crate) txn_id: Option<String>,
    /// All live [`Add`] file actions at this version.
    pub(crate) all_files: Option<Vec<Add>>,
    /// Number of records deleted through Deletion Vectors in this table version.
    pub(crate) num_deleted_records_opt: Option<i64>,
    /// Number of Deletion Vectors active in this table version.
    pub(crate) num_deletion_vectors_opt: Option<i64>,
    /// Distribution of deleted record counts across files.
    pub(crate) deleted_record_counts_histogram_opt: Option<DeletedRecordCountsHistogram>,
}

impl Crc {
    /// Reconstructs CRC state from its in-memory fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the fields do not form semantically consistent CRC state.
    #[internal_api]
    #[cfg_attr(not(feature = "internal-api"), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_from_parts(
        version: Version,
        metadata: Metadata,
        protocol: Protocol,
        file_stats_state: FileStatsState,
        in_commit_timestamp_opt: Option<i64>,
        set_transaction_state: SetTransactionState,
        domain_metadata_state: DomainMetadataState,
        txn_id: Option<String>,
        all_files: Option<Vec<Add>>,
        num_deleted_records_opt: Option<i64>,
        num_deletion_vectors_opt: Option<i64>,
        deleted_record_counts_histogram_opt: Option<DeletedRecordCountsHistogram>,
    ) -> DeltaResult<Self> {
        let crc = Self {
            version,
            metadata,
            protocol,
            file_stats_state,
            in_commit_timestamp_opt,
            set_transaction_state,
            domain_metadata_state,
            txn_id,
            all_files,
            num_deleted_records_opt,
            num_deletion_vectors_opt,
            deleted_record_counts_histogram_opt,
        };
        crc.validate()?;
        Ok(crc)
    }

    /// Returns absolute file-level statistics only if `file_stats_state` is `Complete`.
    ///
    /// Returns `None` when file stats cannot be trusted -- for example, when the CRC was
    /// built from incremental replay that encountered a non-incremental operation or a
    /// missing file size.
    pub fn file_stats(&self) -> Option<&FileStats> {
        self.file_stats_state.file_stats()
    }

    fn validate(&self) -> DeltaResult<()> {
        if let FileStatsState::Complete(stats) = &self.file_stats_state {
            stats.validate()?;
        }
        validate_set_transaction_state(&self.set_transaction_state)?;
        validate_domain_metadata_state(&self.domain_metadata_state)?;

        if let Some(value) = self.num_deleted_records_opt {
            require!(
                value >= 0,
                Error::generic(format!(
                    "CRC has invalid numDeletedRecordsOpt: expected a non-negative value, got {value}"
                ))
            );
        }
        if let Some(value) = self.num_deletion_vectors_opt {
            require!(
                value >= 0,
                Error::generic(format!(
                    "CRC has invalid numDeletionVectorsOpt: expected a non-negative value, got {value}"
                ))
            );
            if let FileStatsState::Complete(stats) = &self.file_stats_state {
                require!(
                    value <= stats.num_files,
                    Error::generic(format!(
                        "CRC numDeletionVectorsOpt {value} exceeds numFiles {}",
                        stats.num_files
                    ))
                );
            }
        }
        if let Some(histogram) = &self.deleted_record_counts_histogram_opt {
            histogram.validate()?;
            if let FileStatsState::Complete(stats) = &self.file_stats_state {
                let histogram_num_files = checked_sum(
                    "deletedRecordCountsHistogramOpt.deletedRecordCounts",
                    &histogram.deleted_record_counts,
                )?;
                require!(
                    histogram_num_files == stats.num_files,
                    Error::generic(format!(
                        "CRC deleted-record histogram file count {histogram_num_files} does not match numFiles {}",
                        stats.num_files
                    ))
                );
            }
        }
        if let Some(all_files) = &self.all_files {
            self.validate_all_files(all_files)?;
        }
        if self.num_deletion_vectors_opt == Some(0) {
            require!(
                self.num_deleted_records_opt.is_none_or(|value| value == 0),
                Error::generic(
                    "CRC numDeletedRecordsOpt must be zero when numDeletionVectorsOpt is zero"
                )
            );
        }

        let properties = self.metadata.parse_table_properties();
        let ict_enabled = self
            .protocol
            .has_table_feature(&TableFeature::InCommitTimestamp)
            && properties.enable_in_commit_timestamps == Some(true);
        require!(
            !ict_enabled || self.in_commit_timestamp_opt.is_some(),
            Error::generic(
                "CRC for an In-Commit-Timestamp-enabled table is missing inCommitTimestampOpt"
            )
        );
        Ok(())
    }

    fn validate_all_files(&self, all_files: &[Add]) -> DeltaResult<()> {
        let num_files = i64::try_from(all_files.len())
            .map_err(|_| Error::generic("allFiles length exceeds the supported range"))?;
        let table_size_bytes = all_files.iter().try_fold(0_i64, |sum, add| {
            require!(
                add.size >= 0,
                Error::generic(format!(
                    "allFiles contains negative file size {} for {}",
                    add.size, add.path
                ))
            );
            sum.checked_add(add.size)
                .ok_or_else(|| Error::generic("allFiles table size overflow"))
        })?;

        if let FileStatsState::Complete(stats) = &self.file_stats_state {
            require!(
                num_files == stats.num_files,
                Error::generic(format!(
                    "CRC allFiles count {num_files} does not match numFiles {}",
                    stats.num_files
                ))
            );
            require!(
                table_size_bytes == stats.table_size_bytes,
                Error::generic(format!(
                    "CRC allFiles byte total {table_size_bytes} does not match tableSizeBytes {}",
                    stats.table_size_bytes
                ))
            );
            if let Some(histogram) = &stats.file_size_histogram {
                histogram.validate_file_sizes(all_files.iter().map(|add| add.size))?;
            }
        }

        let mut num_deletion_vectors = 0_i64;
        let mut num_deleted_records = 0_i64;
        let mut deleted_record_counts = vec![0_i64; DeletedRecordCountsHistogram::NUM_BINS];
        for add in all_files {
            let cardinality = if let Some(deletion_vector) = &add.deletion_vector {
                require!(
                    deletion_vector.cardinality >= 0,
                    Error::generic(format!(
                        "allFiles contains a deletion vector with negative cardinality {}",
                        deletion_vector.cardinality
                    ))
                );
                num_deletion_vectors = num_deletion_vectors
                    .checked_add(1)
                    .ok_or_else(|| Error::generic("allFiles deletion-vector count overflow"))?;
                num_deleted_records = num_deleted_records
                    .checked_add(deletion_vector.cardinality)
                    .ok_or_else(|| Error::generic("allFiles deleted-record count overflow"))?;
                deletion_vector.cardinality
            } else {
                0
            };
            let bin = deleted_record_count_bin(cardinality);
            deleted_record_counts[bin] = deleted_record_counts[bin]
                .checked_add(1)
                .ok_or_else(|| Error::generic("allFiles deleted-record histogram overflow"))?;
        }
        if let Some(expected) = self.num_deletion_vectors_opt {
            require!(
                num_deletion_vectors == expected,
                Error::generic(format!(
                    "CRC allFiles deletion-vector count {num_deletion_vectors} does not match numDeletionVectorsOpt {expected}"
                ))
            );
        }
        if let Some(expected) = self.num_deleted_records_opt {
            require!(
                num_deleted_records == expected,
                Error::generic(format!(
                    "CRC allFiles deleted-record count {num_deleted_records} does not match numDeletedRecordsOpt {expected}"
                ))
            );
        }
        if let Some(expected) = &self.deleted_record_counts_histogram_opt {
            require!(
                deleted_record_counts == expected.deleted_record_counts,
                Error::generic("CRC allFiles does not match deletedRecordCountsHistogramOpt")
            );
        }
        Ok(())
    }

    /// Returns the typed file-stats state. Useful for callers that want to inspect the
    /// variant directly (via `matches!` or the `is_*` predicates).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn file_stats_state(&self) -> &FileStatsState {
        &self.file_stats_state
    }
}

/// Refuses to serialize a degraded (non-`Complete`) CRC, so an invalid state can never
/// round-trip through disk.
impl Serialize for Crc {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CrcRaw::try_from(self)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

// ============================================================================
// CrcRaw: serde intermediate
// ============================================================================

/// The on-disk JSON shape of a CRC file. Serves as the serde intermediate for [`Crc`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CrcRaw {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    txn_id: Option<String>,
    table_size_bytes: i64,
    num_files: i64,
    num_metadata: i64,
    num_protocol: i64,
    metadata: Metadata,
    protocol: Protocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    in_commit_timestamp_opt: Option<i64>,
    #[serde(default)]
    set_transactions: Option<Vec<SetTransaction>>,
    #[serde(default)]
    domain_metadata: Option<Vec<DomainMetadata>>,
    /// The Delta protocol spec names this field `fileSizeHistogram`, but Delta-Spark writers
    /// historically emit it as `histogramOpt`. To remain compatible with CRC files written by
    /// those tools, deserialization accepts either name, but not both. If both are present
    /// deserialization will throw an error. Serialization always emits the spec-correct
    /// `fileSizeHistogram`. Mirrors the kernel-java fix in
    /// <https://github.com/delta-io/delta/pull/6281>.
    #[serde(
        default,
        alias = "histogramOpt",
        deserialize_with = "de_validated_file_size_histogram",
        skip_serializing_if = "Option::is_none"
    )]
    file_size_histogram: Option<FileSizeHistogram>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    all_files: Option<Vec<CrcAddRaw>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_deleted_records_opt: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_deletion_vectors_opt: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deleted_record_counts_histogram_opt: Option<DeletedRecordCountsHistogram>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CrcAddRaw {
    path: String,
    partition_values: HashMap<String, Option<String>>,
    size: i64,
    modification_time: i64,
    data_change: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stats: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tags: Option<HashMap<String, Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deletion_vector: Option<CrcDeletionVectorRaw>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_row_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_row_commit_version: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    clustering_provider: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CrcDeletionVectorRaw {
    storage_type: String,
    path_or_inline_dv: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset: Option<i32>,
    size_in_bytes: i32,
    cardinality: i64,
}

impl CrcAddRaw {
    fn try_into_add(self) -> DeltaResult<Add> {
        let deletion_vector = self
            .deletion_vector
            .map(CrcDeletionVectorRaw::try_into_deletion_vector)
            .transpose()?;
        Ok(Add::from_parts(
            self.path,
            self.partition_values
                .into_iter()
                .filter_map(|(key, value)| value.map(|value| (key, value)))
                .collect(),
            self.size,
            self.modification_time,
            self.data_change,
            self.stats,
            self.tags,
            deletion_vector,
            self.base_row_id,
            self.default_row_commit_version,
            self.clustering_provider,
        ))
    }
}

impl From<&Add> for CrcAddRaw {
    fn from(add: &Add) -> Self {
        Self {
            path: add.path.clone(),
            partition_values: add
                .partition_values
                .iter()
                .map(|(key, value)| (key.clone(), Some(value.clone())))
                .collect(),
            size: add.size,
            modification_time: add.modification_time,
            data_change: add.data_change,
            stats: add.stats.clone(),
            tags: add.tags.clone(),
            deletion_vector: add.deletion_vector.as_ref().map(CrcDeletionVectorRaw::from),
            base_row_id: add.base_row_id,
            default_row_commit_version: add.default_row_commit_version,
            clustering_provider: add.clustering_provider.clone(),
        }
    }
}

impl CrcDeletionVectorRaw {
    fn try_into_deletion_vector(self) -> DeltaResult<DeletionVectorDescriptor> {
        let storage_type: DeletionVectorStorageType = self
            .storage_type
            .parse()
            .map_err(|error: Error| Error::generic(error.to_string()))?;
        DeletionVectorDescriptor::try_new(
            storage_type,
            self.path_or_inline_dv,
            self.offset,
            self.size_in_bytes,
            self.cardinality,
        )
    }
}

impl From<&DeletionVectorDescriptor> for CrcDeletionVectorRaw {
    fn from(deletion_vector: &DeletionVectorDescriptor) -> Self {
        Self {
            storage_type: deletion_vector.storage_type.to_string(),
            path_or_inline_dv: deletion_vector.path_or_inline_dv.clone(),
            offset: deletion_vector.offset,
            size_in_bytes: deletion_vector.size_in_bytes,
            cardinality: deletion_vector.cardinality,
        }
    }
}

impl Crc {
    /// Parses a `.crc` file body for `version`, which comes from the filename because the body does
    /// not carry it.
    ///
    /// Returns parsed CRC state after validating its JSON shape, required action counts,
    /// non-negative aggregate statistics, and histogram structure. This does not compare the state
    /// with log replay.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON or invalid counts, statistics, or histogram fields.
    #[internal_api]
    pub(crate) fn try_from_json_bytes(bytes: &[u8], version: Version) -> DeltaResult<Self> {
        let raw: CrcRaw = serde_json::from_slice(bytes)?;
        // Per the Delta protocol spec, numMetadata and numProtocol MUST be 1 in any CRC file.
        // Reject malformed files at the deserialization boundary so callers can trust the value.
        for (name, value) in [
            ("numMetadata", raw.num_metadata),
            ("numProtocol", raw.num_protocol),
        ] {
            if value != 1 {
                return Err(Error::generic(format!(
                    "CRC file has invalid {name}: expected 1, got {value}"
                )));
            }
        }
        for (name, value) in [
            ("numFiles", raw.num_files),
            ("tableSizeBytes", raw.table_size_bytes),
        ] {
            if value < 0 {
                return Err(Error::generic(format!(
                    "CRC file has invalid {name}: expected a non-negative value, got {value}"
                )));
            }
        }
        // A CRC file on disk is by definition complete; we never deserialize a degraded state.
        let file_stats_state = FileStatsState::Complete(FileStats::try_new(
            raw.num_files,
            raw.table_size_bytes,
            raw.file_size_histogram,
        )?);
        // Present arrays (including empty `[]`) are authoritative. Absent or null arrays leave
        // only partial knowledge.
        let set_transaction_state = match raw.set_transactions {
            Some(values) => SetTransactionState::try_complete(values)?,
            None => SetTransactionState::Partial(HashMap::new()),
        };
        let domain_metadata_state = match raw.domain_metadata {
            Some(values) => DomainMetadataState::try_complete(values)?,
            None => DomainMetadataState::Partial(HashMap::new()),
        };
        let all_files = raw
            .all_files
            .map(|values| {
                values
                    .into_iter()
                    .map(CrcAddRaw::try_into_add)
                    .collect::<DeltaResult<Vec<_>>>()
            })
            .transpose()?;
        let deleted_record_counts_histogram_opt = raw
            .deleted_record_counts_histogram_opt
            .map(|histogram| DeletedRecordCountsHistogram::try_new(histogram.deleted_record_counts))
            .transpose()?;
        Crc::try_from_parts(
            version,
            raw.metadata,
            raw.protocol,
            file_stats_state,
            raw.in_commit_timestamp_opt,
            set_transaction_state,
            domain_metadata_state,
            raw.txn_id,
            all_files,
            raw.num_deleted_records_opt,
            raw.num_deletion_vectors_opt,
            deleted_record_counts_histogram_opt,
        )
    }
}

/// Fails for non-`Complete` file stats: a degraded CRC has no well-defined on-disk shape.
impl TryFrom<&Crc> for CrcRaw {
    type Error = Error;
    fn try_from(crc: &Crc) -> Result<Self, Self::Error> {
        crc.validate()?;
        let FileStatsState::Complete(stats) = &crc.file_stats_state else {
            return Err(Error::ChecksumWriteUnsupported(format!(
                "Cannot serialize CRC with {:?} file stats",
                crc.file_stats_state
            )));
        };
        Ok(CrcRaw {
            txn_id: crc.txn_id.clone(),
            table_size_bytes: stats.table_size_bytes,
            num_files: stats.num_files,
            num_metadata: 1,
            num_protocol: 1,
            metadata: crc.metadata.clone(),
            protocol: crc.protocol.clone(),
            in_commit_timestamp_opt: crc.in_commit_timestamp_opt,
            // Only `Complete` is written; `Partial` is dropped.
            set_transactions: match &crc.set_transaction_state {
                SetTransactionState::Complete(m) => Some(m.values().cloned().collect()),
                SetTransactionState::Partial(_) => None,
            },
            // Only `Complete` is written; `Partial` is dropped.
            domain_metadata: match &crc.domain_metadata_state {
                DomainMetadataState::Complete(m) => Some(m.values().cloned().collect()),
                DomainMetadataState::Partial(_) => None,
            },
            file_size_histogram: stats.file_size_histogram.clone(),
            all_files: crc
                .all_files
                .as_ref()
                .map(|values| values.iter().map(CrcAddRaw::from).collect()),
            num_deleted_records_opt: crc.num_deleted_records_opt,
            num_deletion_vectors_opt: crc.num_deletion_vectors_opt,
            deleted_record_counts_histogram_opt: crc.deleted_record_counts_histogram_opt.clone(),
        })
    }
}

/// Deserializes an `Option<FileSizeHistogram>` from a CRC JSON file with validation.
///
/// After serde deserializes the raw JSON fields, this validates the histogram invariants
/// (sorted boundaries, matching array lengths, etc.) via [`FileSizeHistogram::try_new`],
/// ensuring malformed CRC files are rejected rather than causing panics later.
fn de_validated_file_size_histogram<'de, D>(
    deserializer: D,
) -> Result<Option<FileSizeHistogram>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt: Option<FileSizeHistogram> = Option::deserialize(deserializer)?;
    match opt {
        Some(hist) => {
            if let Some(bin) = hist
                .file_counts
                .iter()
                .zip(&hist.total_bytes)
                .position(|(count, bytes)| *count < 0 || *bytes < 0)
            {
                return Err(serde::de::Error::custom(format!(
                    "CRC fileSizeHistogram has negative counts or bytes at bin {bin}"
                )));
            }
            FileSizeHistogram::try_new(
                hist.sorted_bin_boundaries,
                hist.file_counts,
                hist.total_bytes,
            )
            .map(Some)
            .map_err(serde::de::Error::custom)
        }
        None => Ok(None),
    }
}

/// The [DeletedRecordCountsHistogram] object represents a histogram tracking the distribution of
/// deleted record counts across files in the table. Each bin in the histogram represents a range
/// of deletion counts and stores the number of files having that many deleted records.
///
/// The histogram bins correspond to the following ranges:
/// Bin 0: [0, 0] (files with no deletions)
/// Bin 1: [1, 9] (files with 1-9 deleted records)
/// Bin 2: [10, 99] (files with 10-99 deleted records)
/// Bin 3: [100, 999] (files with 100-999 deleted records)
/// Bin 4: [1000, 9999] (files with 1,000-9,999 deleted records)
/// Bin 5: [10000, 99999] (files with 10,000-99,999 deleted records)
/// Bin 6: [100000, 999999] (files with 100,000-999,999 deleted records)
/// Bin 7: [1000000, 9999999] (files with 1,000,000-9,999,999 deleted records)
/// Bin 8: [10000000, 2147483646] (files with 10,000,000 to 2,147,483,646 deleted records)
/// Bin 9: [2147483647, inf) (files with 2,147,483,647 or more deleted records)
///
/// [DeletedRecordCountsHistogram]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#deleted-record-counts-histogram-schema
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeletedRecordCountsHistogram {
    /// Array of size 10 where each element represents the count of files falling into a specific
    /// deletion count range.
    pub(crate) deleted_record_counts: Vec<i64>,
}

impl DeletedRecordCountsHistogram {
    const NUM_BINS: usize = 10;

    /// Reconstructs a deleted-record-count histogram from its serialized bins after validation.
    ///
    /// # Errors
    ///
    /// Returns an error unless there are exactly ten bins and every file count is non-negative.
    #[internal_api]
    #[cfg_attr(not(feature = "internal-api"), allow(dead_code))]
    pub(crate) fn try_new(deleted_record_counts: Vec<i64>) -> DeltaResult<Self> {
        let histogram = Self {
            deleted_record_counts,
        };
        histogram.validate()?;
        Ok(histogram)
    }

    fn validate(&self) -> DeltaResult<()> {
        require!(
            self.deleted_record_counts.len() == Self::NUM_BINS,
            Error::generic(format!(
                "deleted-record-count histogram must contain exactly {} bins, got {}",
                Self::NUM_BINS,
                self.deleted_record_counts.len()
            ))
        );
        if let Some((bin, count)) = self
            .deleted_record_counts
            .iter()
            .copied()
            .enumerate()
            .find(|(_, count)| *count < 0)
        {
            return Err(Error::generic(format!(
                "deleted-record-count histogram has negative file count {count} at bin {bin}"
            )));
        }
        Ok(())
    }
}

fn checked_sum(name: &str, values: &[i64]) -> DeltaResult<i64> {
    values.iter().try_fold(0_i64, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| Error::generic(format!("CRC {name} sum overflow")))
    })
}

fn deleted_record_count_bin(cardinality: i64) -> usize {
    match cardinality {
        0 => 0,
        1..=9 => 1,
        10..=99 => 2,
        100..=999 => 3,
        1_000..=9_999 => 4,
        10_000..=99_999 => 5,
        100_000..=999_999 => 6,
        1_000_000..=9_999_999 => 7,
        10_000_000..=2_147_483_646 => 8,
        _ => 9,
    }
}

fn validate_set_transaction_state(state: &SetTransactionState) -> DeltaResult<()> {
    let values = match state {
        SetTransactionState::Complete(values) | SetTransactionState::Partial(values) => values,
    };
    for (app_id, transaction) in values {
        require!(
            app_id == &transaction.app_id,
            Error::generic(format!(
                "CRC transaction map key {app_id} does not match action application id {}",
                transaction.app_id
            ))
        );
    }
    Ok(())
}

fn validate_domain_metadata_state(state: &DomainMetadataState) -> DeltaResult<()> {
    let (values, state_name) = match state {
        DomainMetadataState::Complete(values) => (values, "complete"),
        DomainMetadataState::Partial(values) => (values, "partial"),
    };
    for (domain, action) in values {
        require!(
            domain == action.domain(),
            Error::generic(format!(
                "CRC domain metadata map key {domain} does not match action domain {}",
                action.domain()
            ))
        );
        require!(
            !action.is_removed(),
            Error::generic(format!(
                "{state_name} CRC state contains a domain metadata tombstone for {domain}"
            ))
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;

    use super::{
        Crc, CrcRaw, DomainMetadataState, FileSizeHistogram, FileStats, FileStatsState,
        SetTransactionState,
    };
    use crate::actions::{DomainMetadata, Protocol, SetTransaction};
    use crate::table_features::TableFeature;

    /// A minimal valid protocol for round-trip tests. `Protocol::default()` is `(0, 0)`, which
    /// `try_new` rejects, so a default protocol can't round-trip through serde (deserialization
    /// validates via `try_new`).
    fn valid_protocol() -> Protocol {
        Protocol::try_new(1, 1, TableFeature::NO_LIST, TableFeature::NO_LIST).unwrap()
    }

    /// Helper to create a minimal `Crc` with only `set_transaction_state` and
    /// `domain_metadata_state` populated.
    fn crc_with(
        set_transaction_state: SetTransactionState,
        domain_metadata_state: DomainMetadataState,
    ) -> Crc {
        Crc {
            protocol: valid_protocol(),
            set_transaction_state,
            domain_metadata_state,
            ..Default::default()
        }
    }

    #[test]
    fn de_vec_to_map_produces_correct_keys_and_values() {
        let json = r#"{
            "tableSizeBytes": 0,
            "numFiles": 0,
            "numMetadata": 1,
            "numProtocol": 1,
            "metadata": {
                "id": "test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": "{\"type\":\"struct\",\"fields\":[]}",
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0
            },
            "protocol": {"minReaderVersion": 1, "minWriterVersion": 1},
            "setTransactions": [
                {"appId": "app-1", "version": 3, "lastUpdated": 1000},
                {"appId": "app-2", "version": 7}
            ],
            "domainMetadata": [
                {"domain": "delta.rowTracking", "configuration": "{\"rowIdHighWaterMark\":1}", "removed": false},
                {"domain": "delta.clustering", "configuration": "{}", "removed": false}
            ]
        }"#;

        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();

        // A present `setTransactions` array deserializes as `Complete` (authoritative).
        let txns = crc.set_transaction_state.expect_complete();
        assert_eq!(txns.len(), 2);

        let txn1 = &txns["app-1"];
        assert_eq!(txn1.app_id, "app-1");
        assert_eq!(txn1.version, 3);
        assert_eq!(txn1.last_updated, Some(1000));

        let txn2 = &txns["app-2"];
        assert_eq!(txn2.app_id, "app-2");
        assert_eq!(txn2.version, 7);
        assert_eq!(txn2.last_updated, None);

        // A present `domainMetadata` array deserializes as `Complete` (authoritative).
        let domains = crc.domain_metadata_state.expect_complete();
        assert_eq!(domains.len(), 2);
        assert!(domains.contains_key("delta.rowTracking"));
        assert!(domains.contains_key("delta.clustering"));
    }

    #[test]
    fn de_null_dm_and_txns_deserialize_to_partial_empty() {
        let json = r#"{
            "tableSizeBytes": 0,
            "numFiles": 0,
            "numMetadata": 1,
            "numProtocol": 1,
            "metadata": {
                "id": "test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": "{\"type\":\"struct\",\"fields\":[]}",
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0
            },
            "protocol": {"minReaderVersion": 1, "minWriterVersion": 1},
            "setTransactions": null,
            "domainMetadata": null
        }"#;
        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();
        assert_eq!(
            crc.set_transaction_state,
            SetTransactionState::Partial(HashMap::new())
        );
        assert_eq!(
            crc.domain_metadata_state,
            DomainMetadataState::Partial(HashMap::new())
        );
    }

    #[test]
    fn de_missing_dm_and_txns_fields_deserialize_to_partial_empty() {
        let json = r#"{
            "tableSizeBytes": 0,
            "numFiles": 0,
            "numMetadata": 1,
            "numProtocol": 1,
            "metadata": {
                "id": "test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": "{\"type\":\"struct\",\"fields\":[]}",
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0
            },
            "protocol": {"minReaderVersion": 1, "minWriterVersion": 1}
        }"#;
        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();
        assert_eq!(
            crc.set_transaction_state,
            SetTransactionState::Partial(HashMap::new())
        );
        assert_eq!(
            crc.domain_metadata_state,
            DomainMetadataState::Partial(HashMap::new())
        );
    }

    #[test]
    fn ser_partial_dm_and_partial_txns_serialize_to_null() {
        let crc = crc_with(
            SetTransactionState::Partial(HashMap::new()),
            DomainMetadataState::Partial(HashMap::new()),
        );
        let json = serde_json::to_value(&crc).unwrap();
        // Partial is not authoritative for misses; persisting it would falsely promote
        // to `Complete(empty)` on the next read.
        assert!(json["setTransactions"].is_null());
        assert!(json["domainMetadata"].is_null());
    }

    #[test]
    fn ser_non_empty_partial_dm_still_serializes_to_null() {
        let mut partial = HashMap::new();
        partial.insert(
            "delta.rowTracking".to_string(),
            DomainMetadata::new("delta.rowTracking".to_string(), "{}".to_string()),
        );
        let crc = crc_with(
            SetTransactionState::Partial(HashMap::new()),
            DomainMetadataState::Partial(partial),
        );
        let json = serde_json::to_value(&crc).unwrap();
        // Even non-empty Partial maps drop on serialize.
        assert!(json["domainMetadata"].is_null());
    }

    #[test]
    fn ser_non_empty_partial_txns_still_serializes_to_null() {
        let mut partial = HashMap::new();
        partial.insert(
            "my-app".to_string(),
            SetTransaction::new("my-app".to_string(), 1, None),
        );
        let crc = crc_with(
            SetTransactionState::Partial(partial),
            DomainMetadataState::Partial(HashMap::new()),
        );
        let json = serde_json::to_value(&crc).unwrap();
        // Even non-empty Partial maps drop on serialize.
        assert!(json["setTransactions"].is_null());
    }

    #[test]
    fn ser_map_round_trips_through_vec() {
        let mut txns = HashMap::new();
        txns.insert(
            "app-1".to_string(),
            SetTransaction::new("app-1".to_string(), 5, Some(2000)),
        );
        txns.insert(
            "app-2".to_string(),
            SetTransaction::new("app-2".to_string(), 10, None),
        );

        let mut domains = HashMap::new();
        domains.insert(
            "delta.rowTracking".to_string(),
            DomainMetadata::new("delta.rowTracking".to_string(), "{}".to_string()),
        );

        let original = crc_with(
            SetTransactionState::Complete(txns),
            DomainMetadataState::Complete(domains),
        );

        let json_str = serde_json::to_string(&original).unwrap();
        let deserialized = Crc::try_from_json_bytes(json_str.as_bytes(), 0).unwrap();

        assert_eq!(original, deserialized);
    }

    #[test]
    fn round_trip_empty_complete_dm_and_empty_txns() {
        let original = crc_with(
            SetTransactionState::Complete(HashMap::new()),
            DomainMetadataState::Complete(HashMap::new()),
        );

        let json_str = serde_json::to_string(&original).unwrap();
        let deserialized = Crc::try_from_json_bytes(json_str.as_bytes(), 0).unwrap();

        assert_eq!(original, deserialized);

        // Verify the JSON has empty arrays (not null)
        let json_value = serde_json::to_value(&original).unwrap();
        assert_eq!(json_value["setTransactions"], serde_json::json!([]));
        assert_eq!(json_value["domainMetadata"], serde_json::json!([]));
    }

    #[test]
    fn round_trip_partial_dm_becomes_empty_partial() {
        let mut partial = HashMap::new();
        partial.insert(
            "delta.rowTracking".to_string(),
            DomainMetadata::new("delta.rowTracking".to_string(), "{}".to_string()),
        );
        let original = crc_with(
            SetTransactionState::Partial(HashMap::new()),
            DomainMetadataState::Partial(partial),
        );

        let json_str = serde_json::to_string(&original).unwrap();
        let deserialized = Crc::try_from_json_bytes(json_str.as_bytes(), 0).unwrap();

        assert_eq!(
            deserialized.domain_metadata_state,
            DomainMetadataState::Partial(HashMap::new())
        );
    }

    #[test]
    fn partial_txns_written_as_null_reads_back_as_empty_partial() {
        let mut partial = HashMap::new();
        partial.insert(
            "my-app".to_string(),
            SetTransaction::new("my-app".to_string(), 7, Some(1000)),
        );
        let original = crc_with(
            SetTransactionState::Partial(partial),
            DomainMetadataState::Partial(HashMap::new()),
        );

        let json_str = serde_json::to_string(&original).unwrap();
        let deserialized = Crc::try_from_json_bytes(json_str.as_bytes(), 0).unwrap();

        assert_eq!(
            deserialized.set_transaction_state,
            SetTransactionState::Partial(HashMap::new())
        );
    }

    #[test]
    fn test_crc_with_multiple_domain_metadatas_and_set_transactions() {
        let mut txns = HashMap::new();
        txns.insert(
            "streaming-app".to_string(),
            SetTransaction::new("streaming-app".to_string(), 42, Some(1700000000)),
        );
        txns.insert(
            "batch-job".to_string(),
            SetTransaction::new("batch-job".to_string(), 100, None),
        );
        txns.insert(
            "etl-pipeline".to_string(),
            SetTransaction::new("etl-pipeline".to_string(), 7, Some(1700001000)),
        );

        let mut domains = HashMap::new();
        domains.insert(
            "delta.rowTracking".to_string(),
            DomainMetadata::new(
                "delta.rowTracking".to_string(),
                r#"{"rowIdHighWaterMark":500}"#.to_string(),
            ),
        );
        domains.insert(
            "delta.clustering".to_string(),
            DomainMetadata::new("delta.clustering".to_string(), "{}".to_string()),
        );
        domains.insert(
            "custom.app".to_string(),
            DomainMetadata::new("custom.app".to_string(), r#"{"version":"2.0"}"#.to_string()),
        );

        let crc = Crc {
            protocol: valid_protocol(),
            file_stats_state: FileStatsState::Complete(FileStats {
                num_files: 10,
                table_size_bytes: 1024 * 1024,
                file_size_histogram: None,
            }),
            set_transaction_state: SetTransactionState::Complete(txns),
            domain_metadata_state: DomainMetadataState::Complete(domains),
            ..Default::default()
        };

        // Round-trip through JSON
        let json_str = serde_json::to_string(&crc).unwrap();
        let deserialized = Crc::try_from_json_bytes(json_str.as_bytes(), 0).unwrap();

        // Verify scalar fields survive the round-trip
        let stats = deserialized.file_stats().unwrap();
        assert_eq!(stats.table_size_bytes(), 1024 * 1024);
        assert_eq!(stats.num_files(), 10);

        // Verify all set transactions
        let txns = deserialized.set_transaction_state.expect_complete();
        assert_eq!(txns.len(), 3);
        assert_eq!(txns["streaming-app"].version, 42);
        assert_eq!(txns["streaming-app"].last_updated, Some(1700000000));
        assert_eq!(txns["batch-job"].version, 100);
        assert_eq!(txns["batch-job"].last_updated, None);
        assert_eq!(txns["etl-pipeline"].version, 7);

        // Verify all domain metadatas
        let domains = deserialized.domain_metadata_state.expect_complete();
        assert_eq!(domains.len(), 3);
        assert!(domains.contains_key("delta.rowTracking"));
        assert!(domains.contains_key("delta.clustering"));
        assert!(domains.contains_key("custom.app"));
        assert_eq!(
            domains["custom.app"].configuration(),
            r#"{"version":"2.0"}"#
        );

        // Verify the original and deserialized are equal
        assert_eq!(crc, deserialized);
    }

    // ===== numMetadata / numProtocol rejection =====

    /// Minimal CRC JSON with the supplied numMetadata / numProtocol values; used to construct
    /// invalid CRCs and verify rejection.
    fn crc_json_with_counts(
        table_size_bytes: i64,
        num_files: i64,
        num_metadata: i64,
        num_protocol: i64,
    ) -> String {
        format!(
            r#"{{
                "tableSizeBytes": {table_size_bytes},
                "numFiles": {num_files},
                "numMetadata": {num_metadata},
                "numProtocol": {num_protocol},
                "metadata": {{
                    "id": "test",
                    "format": {{"provider": "parquet", "options": {{}}}},
                    "schemaString": "{{\"type\":\"struct\",\"fields\":[]}}",
                    "partitionColumns": [],
                    "configuration": {{}},
                    "createdTime": 0
                }},
                "protocol": {{"minReaderVersion": 1, "minWriterVersion": 1}}
            }}"#
        )
    }

    /// Per the Delta protocol spec, both `numMetadata` and `numProtocol` MUST be 1; any other
    /// value (zero, two, negative) is rejected, and the error names the offending field.
    #[rstest]
    #[case::num_metadata("numMetadata", |b| (b, 1))]
    #[case::num_protocol("numProtocol", |b| (1, b))]
    fn de_invalid_count_is_rejected(
        #[case] field: &str,
        #[case] counts: fn(i64) -> (i64, i64),
        #[values(0i64, 2, 3, -1)] bad: i64,
    ) {
        let (m, p) = counts(bad);
        let json = crc_json_with_counts(0, 0, m, p);
        let err = Crc::try_from_json_bytes(json.as_bytes(), 0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(field),
            "expected error to mention {field} for value {bad}, got: {err}"
        );
    }

    #[rstest]
    #[case::num_files("numFiles", 0, -1)]
    #[case::table_size_bytes("tableSizeBytes", -1, 0)]
    fn de_negative_file_stat_is_rejected(
        #[case] field: &str,
        #[case] table_size_bytes: i64,
        #[case] num_files: i64,
    ) {
        let json = crc_json_with_counts(table_size_bytes, num_files, 1, 1);
        let err = Crc::try_from_json_bytes(json.as_bytes(), 0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(field),
            "expected error to mention {field}: {err}"
        );
    }

    #[rstest]
    #[case::file_count("fileCounts", vec![-1, 0], vec![0, 0])]
    #[case::total_bytes("totalBytes", vec![0, 0], vec![0, -1])]
    fn de_negative_file_size_histogram_stat_is_rejected(
        #[case] field: &str,
        #[case] file_counts: Vec<i64>,
        #[case] total_bytes: Vec<i64>,
    ) {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 0, 1, 1)).unwrap();
        crc["fileSizeHistogram"] = serde_json::json!({
            "sortedBinBoundaries": [0, 1],
            "fileCounts": file_counts,
            "totalBytes": total_bytes,
        });
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains("negative counts or bytes"),
            "expected invalid {field}, got {error}"
        );
        assert!(
            !error.to_string().contains("kernel bug"),
            "malformed external data must not be reported as a kernel bug: {error}"
        );
    }

    #[rstest]
    #[case::file_count(
        1,
        10,
        vec![0, 0],
        vec![0, 10],
        "file count"
    )]
    #[case::byte_total(
        1,
        10,
        vec![1, 0],
        vec![9, 0],
        "byte total"
    )]
    #[case::file_count_overflow(
        0,
        0,
        vec![i64::MAX, 1],
        vec![0, 0],
        "sum overflow"
    )]
    #[case::byte_total_overflow(
        0,
        0,
        vec![0, 0],
        vec![i64::MAX, 1],
        "sum overflow"
    )]
    fn file_stats_reject_histogram_aggregate_errors(
        #[case] num_files: i64,
        #[case] table_size_bytes: i64,
        #[case] file_counts: Vec<i64>,
        #[case] total_bytes: Vec<i64>,
        #[case] expected: &str,
    ) {
        let histogram = FileSizeHistogram::try_new(vec![0, 100], file_counts, total_bytes).unwrap();
        let error = FileStats::try_new(num_files, table_size_bytes, Some(histogram)).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[rstest]
    #[case::transactions(
        "setTransactions",
        serde_json::json!([
            {"appId": "app", "version": 1},
            {"appId": "app", "version": 2}
        ]),
        "duplicate transaction"
    )]
    #[case::domains(
        "domainMetadata",
        serde_json::json!([
            {"domain": "domain", "configuration": "{}", "removed": false},
            {"domain": "domain", "configuration": "{}", "removed": false}
        ]),
        "duplicate domain metadata"
    )]
    #[case::domain_tombstone(
        "domainMetadata",
        serde_json::json!([
            {"domain": "domain", "configuration": "{}", "removed": true}
        ]),
        "tombstone"
    )]
    fn de_rejects_duplicate_or_removed_map_state(
        #[case] field: &str,
        #[case] value: serde_json::Value,
        #[case] expected: &str,
    ) {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 0, 1, 1)).unwrap();
        crc[field] = value;
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[rstest]
    #[case::too_few(vec![0; 9], "exactly 10 bins")]
    #[case::too_many(vec![0; 11], "exactly 10 bins")]
    #[case::negative(vec![0, 0, 0, -1, 0, 0, 0, 0, 0, 0], "negative file count")]
    #[case::sum_overflow(
        vec![i64::MAX, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "sum overflow"
    )]
    fn de_rejects_invalid_deleted_record_histogram(
        #[case] counts: Vec<i64>,
        #[case] expected: &str,
    ) {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 0, 1, 1)).unwrap();
        crc["deletedRecordCountsHistogramOpt"] = serde_json::json!({"deletedRecordCounts": counts});
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[rstest]
    #[case::deleted_records("numDeletedRecordsOpt")]
    #[case::deletion_vectors("numDeletionVectorsOpt")]
    fn de_rejects_negative_deletion_vector_aggregate(#[case] field: &str) {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 0, 1, 1)).unwrap();
        crc[field] = serde_json::json!(-1);
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains(field),
            "unexpected error: {error}"
        );
    }

    fn crc_json_with_one_file() -> serde_json::Value {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(50, 1, 1, 1)).unwrap();
        crc["fileSizeHistogram"] = serde_json::json!({
            "sortedBinBoundaries": [0, 100],
            "fileCounts": [1, 0],
            "totalBytes": [50, 0]
        });
        crc["allFiles"] = serde_json::json!([{
            "path": "part.parquet",
            "partitionValues": {"nullable_partition": null},
            "size": 50,
            "modificationTime": 1,
            "dataChange": false
        }]);
        crc["numDeletedRecordsOpt"] = serde_json::json!(0);
        crc["numDeletionVectorsOpt"] = serde_json::json!(0);
        crc["deletedRecordCountsHistogramOpt"] = serde_json::json!({
            "deletedRecordCounts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        });
        crc
    }

    #[test]
    fn de_full_file_state_round_trips_through_checked_crc() {
        let json = crc_json_with_one_file();
        let crc = Crc::try_from_json_bytes(json.to_string().as_bytes(), 7).unwrap();
        assert_eq!(crc.version, 7);
        assert_eq!(crc.all_files.as_ref().unwrap().len(), 1);
        assert!(crc.all_files.as_ref().unwrap()[0]
            .partition_values
            .is_empty());

        let round_tripped = serde_json::to_vec(&crc).unwrap();
        assert_eq!(Crc::try_from_json_bytes(&round_tripped, 7).unwrap(), crc);
    }

    #[rstest]
    #[case::file_count("numFiles", serde_json::json!(2), "allFiles count")]
    #[case::table_size("tableSizeBytes", serde_json::json!(51), "allFiles byte total")]
    fn de_rejects_all_files_top_level_mismatch(
        #[case] field: &str,
        #[case] value: serde_json::Value,
        #[case] expected: &str,
    ) {
        let mut crc = crc_json_with_one_file();
        crc.as_object_mut().unwrap().remove("fileSizeHistogram");
        crc.as_object_mut()
            .unwrap()
            .remove("deletedRecordCountsHistogramOpt");
        crc[field] = value;
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn de_rejects_all_files_histogram_bin_mismatch() {
        let mut crc = crc_json_with_one_file();
        crc["fileSizeHistogram"]["fileCounts"] = serde_json::json!([0, 1]);
        crc["fileSizeHistogram"]["totalBytes"] = serde_json::json!([0, 50]);
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(error.to_string().contains("allFiles"));
    }

    #[test]
    fn de_rejects_all_files_table_size_overflow() {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(i64::MAX, 2, 1, 1)).unwrap();
        crc["allFiles"] = serde_json::json!([
            {"path": "a", "partitionValues": {}, "size": i64::MAX,
             "modificationTime": 1, "dataChange": false},
            {"path": "b", "partitionValues": {}, "size": 1,
             "modificationTime": 1, "dataChange": false}
        ]);
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(error.to_string().contains("table size overflow"));
    }

    fn add_deletion_vector(crc: &mut serde_json::Value, cardinality: i64) {
        crc["allFiles"][0]["deletionVector"] = serde_json::json!({
            "storageType": "p",
            "pathOrInlineDv": "file:///dv.bin",
            "offset": 0,
            "sizeInBytes": 1,
            "cardinality": cardinality
        });
    }

    #[rstest]
    #[case::record_count("numDeletedRecordsOpt", serde_json::json!(12), "deleted-record count")]
    #[case::vector_count("numDeletionVectorsOpt", serde_json::json!(0), "deletion-vector count")]
    #[case::histogram(
        "deletedRecordCountsHistogramOpt",
        serde_json::json!({"deletedRecordCounts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}),
        "deletedRecordCountsHistogramOpt"
    )]
    fn de_rejects_all_files_deletion_vector_mismatch(
        #[case] field: &str,
        #[case] value: serde_json::Value,
        #[case] expected: &str,
    ) {
        let mut crc = crc_json_with_one_file();
        add_deletion_vector(&mut crc, 13);
        crc["numDeletedRecordsOpt"] = serde_json::json!(13);
        crc["numDeletionVectorsOpt"] = serde_json::json!(1);
        crc["deletedRecordCountsHistogramOpt"] = serde_json::json!({
            "deletedRecordCounts": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        });
        crc[field] = value;
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn de_rejects_all_files_deleted_record_overflow() {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 2, 1, 1)).unwrap();
        crc["allFiles"] = serde_json::json!([
            {"path": "a", "partitionValues": {}, "size": 0,
             "modificationTime": 1, "dataChange": false},
            {"path": "b", "partitionValues": {}, "size": 0,
             "modificationTime": 1, "dataChange": false}
        ]);
        add_deletion_vector(&mut crc, i64::MAX);
        crc["allFiles"][1]["deletionVector"] = crc["allFiles"][0]["deletionVector"].clone();
        crc["allFiles"][1]["deletionVector"]["cardinality"] = serde_json::json!(1);
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(error.to_string().contains("deleted-record count overflow"));
    }

    #[test]
    fn de_requires_ict_when_table_has_ict_enabled() {
        let mut crc: serde_json::Value =
            serde_json::from_str(&crc_json_with_counts(0, 0, 1, 1)).unwrap();
        crc["metadata"]["configuration"] =
            serde_json::json!({"delta.enableInCommitTimestamps": "true"});
        crc["protocol"] = serde_json::json!({
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": [],
            "writerFeatures": ["inCommitTimestamp"]
        });
        let error = Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap_err();
        assert!(error.to_string().contains("inCommitTimestampOpt"));

        crc["inCommitTimestampOpt"] = serde_json::json!(1234);
        Crc::try_from_json_bytes(crc.to_string().as_bytes(), 0).unwrap();
    }

    // ===== protocol validation on the CRC deserialization path =====

    /// Minimal CRC JSON whose `protocol` is the supplied fragment. Proves CRC deserialization
    /// runs the protocol through `Protocol::try_new` instead of building an unchecked one.
    fn crc_json_with_protocol(protocol: &str) -> String {
        format!(
            r#"{{
                "tableSizeBytes": 0,
                "numFiles": 0,
                "numMetadata": 1,
                "numProtocol": 1,
                "metadata": {{
                    "id": "test",
                    "format": {{"provider": "parquet", "options": {{}}}},
                    "schemaString": "{{\"type\":\"struct\",\"fields\":[]}}",
                    "partitionColumns": [],
                    "configuration": {{}},
                    "createdTime": 0
                }},
                "protocol": {protocol}
            }}"#
        )
    }

    #[test]
    fn deserialize_crc_accepts_orphaned_column_mapping() {
        let json = crc_json_with_protocol(
            r#"{"minReaderVersion": 3, "minWriterVersion": 7,
                "readerFeatures": [], "writerFeatures": ["columnMapping"]}"#,
        );
        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();
        assert_eq!(crc.protocol.min_reader_version(), 3);
    }

    #[test]
    fn deserialize_crc_rejects_orphaned_non_legacy_reader_writer_feature() {
        let json = crc_json_with_protocol(
            r#"{"minReaderVersion": 3, "minWriterVersion": 7,
                "readerFeatures": [], "writerFeatures": ["columnMapping", "deletionVectors"]}"#,
        );
        assert!(Crc::try_from_json_bytes(json.as_bytes(), 0).is_err());
    }

    #[test]
    fn ser_indeterminate_file_stats_returns_error() {
        let crc = Crc {
            file_stats_state: FileStatsState::Indeterminate,
            ..Default::default()
        };
        let err = serde_json::to_string(&crc).unwrap_err().to_string();
        assert!(
            err.contains("Cannot serialize CRC"),
            "expected serialize-rejection error, got: {err}"
        );
    }

    #[test]
    fn try_from_ref_indeterminate_returns_checksum_write_unsupported() {
        let crc = Crc {
            file_stats_state: FileStatsState::Indeterminate,
            ..Default::default()
        };
        let err = CrcRaw::try_from(&crc).unwrap_err();
        assert!(
            matches!(err, crate::Error::ChecksumWriteUnsupported(_)),
            "expected ChecksumWriteUnsupported, got: {err:?}"
        );
    }

    // ===== File size histogram validation =====

    /// Minimal CRC JSON with a file size histogram field spliced in under the given field name
    /// (`fileSizeHistogram` per the Delta spec, or `histogramOpt` for legacy Delta-Spark
    /// compatibility).
    fn crc_json_with_histogram(field_name: &str, histogram_json: &str) -> String {
        format!(
            r#"{{
                "tableSizeBytes": 0,
                "numFiles": 0,
                "numMetadata": 1,
                "numProtocol": 1,
                "metadata": {{
                    "id": "test",
                    "format": {{"provider": "parquet", "options": {{}}}},
                    "schemaString": "{{\"type\":\"struct\",\"fields\":[]}}",
                    "partitionColumns": [],
                    "configuration": {{}},
                    "createdTime": 0
                }},
                "protocol": {{"minReaderVersion": 1, "minWriterVersion": 1}},
                "{field_name}": {histogram_json}
            }}"#
        )
    }

    /// Both the Delta spec field name and the legacy Delta-Spark name must deserialize.
    #[rstest]
    #[case::spec_name("fileSizeHistogram")]
    #[case::legacy_name("histogramOpt")]
    fn de_valid_file_size_histogram_succeeds(#[case] field_name: &str) {
        let json = crc_json_with_histogram(
            field_name,
            r#"{"sortedBinBoundaries": [0, 100, 200], "fileCounts": [0, 0, 0], "totalBytes": [0, 0, 0]}"#,
        );
        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();
        assert!(crc.file_stats().unwrap().file_size_histogram().is_some());
    }

    #[rstest]
    #[case::spec_name("fileSizeHistogram")]
    #[case::legacy_name("histogramOpt")]
    fn de_null_file_size_histogram_deserializes_to_none(#[case] field_name: &str) {
        let json = crc_json_with_histogram(field_name, "null");
        let crc = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap();
        assert!(crc.file_stats().unwrap().file_size_histogram().is_none());
    }

    /// Validation must reject malformed histograms regardless of which field name they arrived
    /// under. Cartesian product across malformed payloads x both accepted field names.
    #[rstest]
    #[case::unsorted_boundaries(
        r#"{"sortedBinBoundaries": [0, 200, 100], "fileCounts": [0, 0, 0], "totalBytes": [0, 0, 0]}"#
    )]
    #[case::nonzero_first_boundary(
        r#"{"sortedBinBoundaries": [1, 100], "fileCounts": [0, 0], "totalBytes": [0, 0]}"#
    )]
    #[case::mismatched_lengths(
        r#"{"sortedBinBoundaries": [0, 100], "fileCounts": [0], "totalBytes": [0, 0]}"#
    )]
    #[case::single_boundary(
        r#"{"sortedBinBoundaries": [0], "fileCounts": [0], "totalBytes": [0]}"#
    )]
    fn de_malformed_file_size_histogram_returns_error(
        #[case] histogram_json: &str,
        #[values("fileSizeHistogram", "histogramOpt")] field_name: &str,
    ) {
        let json = crc_json_with_histogram(field_name, histogram_json);
        assert!(Crc::try_from_json_bytes(json.as_bytes(), 0).is_err());
    }

    /// CRC files written by kernel always use the spec-correct field name `fileSizeHistogram`,
    /// even when the input JSON used the legacy `histogramOpt` alias. This matches kernel-java
    /// (delta-io/delta#6281) and ensures kernel-written CRCs are protocol-compliant.
    #[test]
    fn ser_uses_spec_field_name_after_deserializing_legacy_alias() {
        let legacy_json = crc_json_with_histogram(
            "histogramOpt",
            r#"{"sortedBinBoundaries": [0, 100], "fileCounts": [0, 0], "totalBytes": [0, 0]}"#,
        );
        let crc = Crc::try_from_json_bytes(legacy_json.as_bytes(), 0).unwrap();

        let serialized = serde_json::to_value(&crc).unwrap();
        assert!(serialized.get("fileSizeHistogram").is_some());
        assert!(serialized.get("histogramOpt").is_none());
    }

    /// A CRC that contains both `fileSizeHistogram` and `histogramOpt` is rejected with a
    /// "duplicate field" error -- serde's `#[serde(alias)]` treats both names as the same
    /// logical field and refuses to deserialize repeated sets. No real producer emits both
    /// fields today (Delta-Spark writes only `histogramOpt` or only `fileSizeHistogram`,
    /// kernel-java / kernel-rust write only `fileSizeHistogram`), so this is a defensive guard
    /// against malformed CRCs. The error fires regardless of the data carried under each name.
    /// The cases below exercise both matching and mismatched payloads, in both JSON orderings.
    #[rstest]
    #[case::matching_payloads(
        r#"{"sortedBinBoundaries": [0, 100], "fileCounts": [1, 0], "totalBytes": [50, 0]}"#,
        r#"{"sortedBinBoundaries": [0, 100], "fileCounts": [1, 0], "totalBytes": [50, 0]}"#
    )]
    #[case::mismatched_payloads(
        r#"{"sortedBinBoundaries": [0, 100], "fileCounts": [1, 0], "totalBytes": [50, 0]}"#,
        r#"{"sortedBinBoundaries": [0, 200], "fileCounts": [9, 9], "totalBytes": [99, 99]}"#
    )]
    fn de_both_field_names_present_returns_duplicate_field_error(
        #[case] histogram_opt_payload: &str,
        #[case] file_size_histogram_payload: &str,
        #[values(true, false)] spec_listed_last: bool,
    ) {
        let (first_name, first_payload, second_name, second_payload) = if spec_listed_last {
            (
                "histogramOpt",
                histogram_opt_payload,
                "fileSizeHistogram",
                file_size_histogram_payload,
            )
        } else {
            (
                "fileSizeHistogram",
                file_size_histogram_payload,
                "histogramOpt",
                histogram_opt_payload,
            )
        };
        let json = format!(
            r#"{{
                "tableSizeBytes": 0,
                "numFiles": 0,
                "numMetadata": 1,
                "numProtocol": 1,
                "metadata": {{
                    "id": "test",
                    "format": {{"provider": "parquet", "options": {{}}}},
                    "schemaString": "{{\"type\":\"struct\",\"fields\":[]}}",
                    "partitionColumns": [],
                    "configuration": {{}},
                    "createdTime": 0
                }},
                "protocol": {{"minReaderVersion": 1, "minWriterVersion": 1}},
                "{first_name}": {first_payload},
                "{second_name}": {second_payload}
            }}"#
        );
        let err = Crc::try_from_json_bytes(json.as_bytes(), 0).unwrap_err();
        assert!(
            err.to_string().contains("duplicate field"),
            "expected duplicate-field error, got: {err}"
        );
    }
}
