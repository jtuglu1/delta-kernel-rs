//! Typed FFI construction of connector-provided snapshot hints.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::actions::{
    CheckpointMetadata, DomainMetadata, Metadata, Protocol, SetTransaction, Sidecar,
};
use delta_kernel::crc::{Crc, FileSizeHistogram};
use delta_kernel::last_checkpoint_hint::{HintAction, LastCheckpointHint, LastCheckpointV2};
use delta_kernel::snapshot::{SnapshotHint, SnapshotHintVersionStatus};
use delta_kernel::{DeltaResult, Error, Version};

use crate::error::{ExternResult, IntoExternResult};
use crate::handle::Handle;
use crate::log_path::LogPathArray;
use crate::{FfiSnapshotBuilder, KernelStringSlice, MutableFfiSnapshotBuilder, TryFromStringSlice};

/// Integer freshness claim attached to a connector-provided snapshot hint.
pub type FfiSnapshotHintVersionStatus = u32;

/// The connector has not established that the hinted version is latest.
pub const SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED: FfiSnapshotHintVersionStatus = 0;

/// The connector has established that the hinted version is latest.
pub const SNAPSHOT_HINT_VERSION_STATUS_LATEST: FfiSnapshotHintVersionStatus = 1;

/// Borrowed optional UTF-8 string.
#[repr(C)]
pub struct FfiOptionalString {
    /// Whether `value` is present.
    pub has_value: bool,
    /// Borrowed string value. Ignored when `has_value` is false.
    pub value: KernelStringSlice,
}

/// Borrowed optional signed 64-bit integer.
#[repr(C)]
pub struct FfiOptionalI64 {
    /// Whether `value` is present.
    pub has_value: bool,
    /// Integer value. Ignored when `has_value` is false.
    pub value: i64,
}

/// Borrowed optional unsigned 64-bit integer.
#[repr(C)]
pub struct FfiOptionalU64 {
    /// Whether `value` is present.
    pub has_value: bool,
    /// Integer value. Ignored when `has_value` is false.
    pub value: u64,
}

/// Borrowed array of UTF-8 strings.
#[repr(C)]
pub struct FfiStringArray {
    /// Pointer to `len` string slices, or null when `len` is zero.
    pub ptr: *const KernelStringSlice,
    /// Number of strings in the array.
    pub len: usize,
}

/// Borrowed optional array of UTF-8 strings.
#[repr(C)]
pub struct FfiOptionalStringArray {
    /// Whether the array is present. A present empty array differs from an absent array.
    pub has_value: bool,
    /// Borrowed array value. Ignored when `has_value` is false.
    pub value: FfiStringArray,
}

/// One borrowed UTF-8 map entry.
#[repr(C)]
pub struct FfiStringMapEntry {
    /// Entry key.
    pub key: KernelStringSlice,
    /// Entry value.
    pub value: KernelStringSlice,
}

/// Borrowed array of UTF-8 map entries.
#[repr(C)]
pub struct FfiStringMap {
    /// Pointer to `len` entries, or null when `len` is zero.
    pub ptr: *const FfiStringMapEntry,
    /// Number of entries.
    pub len: usize,
}

/// Borrowed optional UTF-8 map.
#[repr(C)]
pub struct FfiOptionalStringMap {
    /// Whether the map is present. A present empty map differs from an absent map.
    pub has_value: bool,
    /// Borrowed map value. Ignored when `has_value` is false.
    pub value: FfiStringMap,
}

/// Borrowed protocol state copied by the protocol setter.
#[repr(C)]
pub struct FfiSnapshotHintProtocol {
    /// Minimum reader protocol version.
    pub min_reader_version: i32,
    /// Minimum writer protocol version.
    pub min_writer_version: i32,
    /// Optional reader feature list.
    pub reader_features: FfiOptionalStringArray,
    /// Optional writer feature list.
    pub writer_features: FfiOptionalStringArray,
}

/// Borrowed metadata state copied by the metadata setter.
#[repr(C)]
pub struct FfiSnapshotHintMetadata {
    /// Table identifier.
    pub id: KernelStringSlice,
    /// Optional table name.
    pub name: FfiOptionalString,
    /// Optional table description.
    pub description: FfiOptionalString,
    /// Data format provider.
    pub format_provider: KernelStringSlice,
    /// Data format options.
    pub format_options: FfiStringMap,
    /// Canonical Delta schema string.
    pub schema_string: KernelStringSlice,
    /// Logical partition column names.
    pub partition_columns: FfiStringArray,
    /// Optional metadata creation time in milliseconds since the Unix epoch.
    pub created_time: FfiOptionalI64,
    /// Table configuration entries.
    pub configuration: FfiStringMap,
}

/// Typed set-transaction action used by a snapshot hint CRC or V2 checkpoint hint.
#[repr(C)]
pub struct FfiSnapshotHintSetTransaction {
    /// Application identifier.
    pub app_id: KernelStringSlice,
    /// Application-specific transaction version.
    pub version: i64,
    /// Optional last-updated time in milliseconds since the Unix epoch.
    pub last_updated: FfiOptionalI64,
}

/// Typed domain-metadata action used by a snapshot hint CRC or V2 checkpoint hint.
#[repr(C)]
pub struct FfiSnapshotHintDomainMetadata {
    /// Domain identifier.
    pub domain: KernelStringSlice,
    /// Domain configuration payload.
    pub configuration: KernelStringSlice,
    /// Whether this action removes the domain.
    pub removed: bool,
}

/// Typed checkpoint-metadata action used by a V2 checkpoint hint.
#[repr(C)]
pub struct FfiSnapshotHintCheckpointMetadata {
    /// Checkpoint version.
    pub version: i64,
    /// Optional action tags.
    pub tags: FfiOptionalStringMap,
}

/// Typed V2 checkpoint sidecar.
#[repr(C)]
pub struct FfiSnapshotHintSidecar {
    /// Sidecar path.
    pub path: KernelStringSlice,
    /// Sidecar size in bytes.
    pub size_in_bytes: i64,
    /// Sidecar modification time in milliseconds since the Unix epoch.
    pub modification_time: i64,
    /// Optional sidecar tags.
    pub tags: FfiOptionalStringMap,
}

/// Integer discriminator for a V2 checkpoint non-file action.
pub type FfiSnapshotHintActionKind = u32;

/// Metadata action discriminator.
pub const SNAPSHOT_HINT_ACTION_METADATA: FfiSnapshotHintActionKind = 0;
/// Protocol action discriminator.
pub const SNAPSHOT_HINT_ACTION_PROTOCOL: FfiSnapshotHintActionKind = 1;
/// Set-transaction action discriminator.
pub const SNAPSHOT_HINT_ACTION_TRANSACTION: FfiSnapshotHintActionKind = 2;
/// Domain-metadata action discriminator.
pub const SNAPSHOT_HINT_ACTION_DOMAIN_METADATA: FfiSnapshotHintActionKind = 3;
/// Checkpoint-metadata action discriminator.
pub const SNAPSHOT_HINT_ACTION_CHECKPOINT_METADATA: FfiSnapshotHintActionKind = 4;

/// Typed payload pointer for a V2 checkpoint non-file action.
#[repr(C)]
#[derive(Clone, Copy)]
pub union FfiSnapshotHintActionValue {
    /// Metadata payload.
    pub metadata: *const FfiSnapshotHintMetadata,
    /// Protocol payload.
    pub protocol: *const FfiSnapshotHintProtocol,
    /// Set-transaction payload.
    pub transaction: *const FfiSnapshotHintSetTransaction,
    /// Domain-metadata payload.
    pub domain_metadata: *const FfiSnapshotHintDomainMetadata,
    /// Checkpoint-metadata payload.
    pub checkpoint_metadata: *const FfiSnapshotHintCheckpointMetadata,
}

/// One typed V2 checkpoint non-file action.
#[repr(C)]
pub struct FfiSnapshotHintAction {
    /// Type of the object addressed by `value`.
    pub kind: FfiSnapshotHintActionKind,
    /// Corresponding typed payload pointer, valid for the duration of the setter call.
    pub value: FfiSnapshotHintActionValue,
}

/// Borrowed array of typed V2 checkpoint sidecars.
#[repr(C)]
pub struct FfiSnapshotHintSidecarArray {
    /// Pointer to `len` sidecars, or null when `len` is zero.
    pub ptr: *const FfiSnapshotHintSidecar,
    /// Number of sidecars.
    pub len: usize,
}

/// Borrowed array of typed V2 checkpoint non-file actions.
#[repr(C)]
pub struct FfiSnapshotHintActionArray {
    /// Pointer to `len` actions, or null when `len` is zero.
    pub ptr: *const FfiSnapshotHintAction,
    /// Number of actions.
    pub len: usize,
}

/// Typed V2 checkpoint fields.
#[repr(C)]
pub struct FfiSnapshotHintV2Checkpoint {
    /// Checkpoint file name.
    pub path: KernelStringSlice,
    /// Optional checkpoint file size.
    pub size_in_bytes: FfiOptionalI64,
    /// Optional checkpoint file modification time.
    pub modification_time: FfiOptionalI64,
    /// Whether sidecar information is present.
    pub has_sidecar_files: bool,
    /// Sidecars. Ignored when `has_sidecar_files` is false.
    pub sidecar_files: FfiSnapshotHintSidecarArray,
    /// Whether non-file actions are present.
    pub has_non_file_actions: bool,
    /// Non-file actions. Ignored when `has_non_file_actions` is false.
    pub non_file_actions: FfiSnapshotHintActionArray,
}

/// Typed `_last_checkpoint` fields.
#[repr(C)]
pub struct FfiSnapshotHintLastCheckpoint {
    /// Checkpoint version.
    pub version: Version,
    /// Number of actions in the checkpoint.
    pub size: i64,
    /// Optional number of checkpoint parts.
    pub parts: FfiOptionalU64,
    /// Optional total checkpoint size in bytes.
    pub size_in_bytes: FfiOptionalI64,
    /// Optional number of Add actions.
    pub num_of_add_files: FfiOptionalI64,
    /// Optional canonical checkpoint schema string.
    pub checkpoint_schema: FfiOptionalString,
    /// Optional checkpoint JSON checksum.
    pub checksum: FfiOptionalString,
    /// Optional checkpoint tags.
    pub tags: FfiOptionalStringMap,
    /// Optional typed V2 checkpoint information.
    pub v2_checkpoint: *const FfiSnapshotHintV2Checkpoint,
}

/// Borrowed array of signed 64-bit integers.
#[repr(C)]
pub struct FfiI64Array {
    /// Pointer to `len` integers, or null when `len` is zero.
    pub ptr: *const i64,
    /// Number of integers.
    pub len: usize,
}

/// Typed file-size histogram fields.
#[repr(C)]
pub struct FfiSnapshotHintFileSizeHistogram {
    /// Sorted lower boundary of every histogram bin.
    pub sorted_bin_boundaries: FfiI64Array,
    /// File count in every histogram bin.
    pub file_counts: FfiI64Array,
    /// Total bytes in every histogram bin.
    pub total_bytes: FfiI64Array,
}

/// Borrowed array of typed set-transaction actions.
#[repr(C)]
pub struct FfiSnapshotHintSetTransactionArray {
    /// Pointer to `len` actions, or null when `len` is zero.
    pub ptr: *const FfiSnapshotHintSetTransaction,
    /// Number of actions.
    pub len: usize,
}

/// Borrowed array of typed domain-metadata actions.
#[repr(C)]
pub struct FfiSnapshotHintDomainMetadataArray {
    /// Pointer to `len` actions, or null when `len` is zero.
    pub ptr: *const FfiSnapshotHintDomainMetadata,
    /// Number of actions.
    pub len: usize,
}

/// Typed CRC fields for a snapshot hint.
#[repr(C)]
pub struct FfiSnapshotHintCrc {
    /// Total table size in bytes.
    pub table_size_bytes: i64,
    /// Number of active files.
    pub num_files: i64,
    /// Optional in-commit timestamp.
    pub in_commit_timestamp: FfiOptionalI64,
    /// Optional file-size histogram.
    pub file_size_histogram: *const FfiSnapshotHintFileSizeHistogram,
    /// Whether the transaction list is present and complete.
    pub has_set_transactions: bool,
    /// Set transactions. Ignored when `has_set_transactions` is false.
    pub set_transactions: FfiSnapshotHintSetTransactionArray,
    /// Whether the domain-metadata list is present and complete.
    pub has_domain_metadata: bool,
    /// Domain metadata. Ignored when `has_domain_metadata` is false.
    pub domain_metadata: FfiSnapshotHintDomainMetadataArray,
}

pub(crate) struct SnapshotHintVisitorState {
    version: Version,
    version_status: SnapshotHintVersionStatus,
    log_paths: Option<Vec<delta_kernel::LogPath>>,
    protocol: Option<Protocol>,
    metadata: Option<Metadata>,
    last_checkpoint_hint: Option<LastCheckpointHint>,
    crc: Option<Arc<Crc>>,
}

pub(crate) enum FfiSnapshotHintState {
    None,
    Building(Box<SnapshotHintVisitorState>),
    Ready(Box<SnapshotHint>),
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSnapshotHint(message.into())
}

fn invalid_from_error(error: Error) -> Error {
    let message = match error {
        Error::Backtraced { source, .. } => return invalid_from_error(*source),
        Error::Generic(message)
        | Error::InternalError(message)
        | Error::InvalidCheckpoint(message)
        | Error::InvalidLogPath(message)
        | Error::InvalidSnapshotHint(message) => message,
        error => error.to_string(),
    };
    invalid(message)
}

fn parse_version_status(
    value: FfiSnapshotHintVersionStatus,
) -> DeltaResult<SnapshotHintVersionStatus> {
    match value {
        SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED => Ok(SnapshotHintVersionStatus::Unverified),
        SNAPSHOT_HINT_VERSION_STATUS_LATEST => Ok(SnapshotHintVersionStatus::Latest),
        value => Err(invalid(format!(
            "unknown snapshot hint version status: {value}"
        ))),
    }
}

fn optional_value<T>(
    has_value: bool,
    value: impl FnOnce() -> DeltaResult<T>,
) -> DeltaResult<Option<T>> {
    has_value.then(value).transpose()
}

unsafe fn string(value: &KernelStringSlice) -> DeltaResult<String> {
    let value: &str = unsafe { TryFromStringSlice::try_from_slice(value) }?;
    Ok(value.to_string())
}

unsafe fn optional_string(value: &FfiOptionalString) -> DeltaResult<Option<String>> {
    optional_value(value.has_value, || unsafe { string(&value.value) })
}

fn optional_i64(value: &FfiOptionalI64) -> Option<i64> {
    value.has_value.then_some(value.value)
}

fn optional_usize(value: &FfiOptionalU64) -> DeltaResult<Option<usize>> {
    optional_value(value.has_value, || {
        usize::try_from(value.value).map_err(|_| {
            invalid(format!(
                "checkpoint part count overflows usize: {}",
                value.value
            ))
        })
    })
}

unsafe fn raw_slice<'a, T>(ptr: *const T, len: usize, name: &str) -> DeltaResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(invalid(format!("{name} pointer is null with length {len}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn required_ref<'a, T>(ptr: *const T, name: &str) -> DeltaResult<&'a T> {
    unsafe { ptr.as_ref() }.ok_or_else(|| invalid(format!("snapshot hint {name} value is null")))
}

unsafe fn strings(value: &FfiStringArray) -> DeltaResult<Vec<String>> {
    unsafe { raw_slice(value.ptr, value.len, "string array") }?
        .iter()
        .map(|value| unsafe { string(value) })
        .collect()
}

unsafe fn optional_strings(value: &FfiOptionalStringArray) -> DeltaResult<Option<Vec<String>>> {
    optional_value(value.has_value, || unsafe { strings(&value.value) })
}

unsafe fn string_map(value: &FfiStringMap) -> DeltaResult<HashMap<String, String>> {
    unsafe { raw_slice(value.ptr, value.len, "string map") }?
        .iter()
        .map(|entry| {
            Ok((unsafe { string(&entry.key) }?, unsafe {
                string(&entry.value)
            }?))
        })
        .collect()
}

unsafe fn optional_string_map(
    value: &FfiOptionalStringMap,
) -> DeltaResult<Option<HashMap<String, String>>> {
    optional_value(value.has_value, || unsafe { string_map(&value.value) })
}

unsafe fn protocol(value: &FfiSnapshotHintProtocol) -> DeltaResult<Protocol> {
    Protocol::try_new(
        value.min_reader_version,
        value.min_writer_version,
        unsafe { optional_strings(&value.reader_features) }?,
        unsafe { optional_strings(&value.writer_features) }?,
    )
}

unsafe fn metadata(value: &FfiSnapshotHintMetadata) -> DeltaResult<Metadata> {
    Ok(Metadata::from_parts(
        unsafe { string(&value.id) }?,
        unsafe { optional_string(&value.name) }?,
        unsafe { optional_string(&value.description) }?,
        unsafe { string(&value.format_provider) }?,
        unsafe { string_map(&value.format_options) }?,
        unsafe { string(&value.schema_string) }?,
        unsafe { strings(&value.partition_columns) }?,
        optional_i64(&value.created_time),
        unsafe { string_map(&value.configuration) }?,
    ))
}

unsafe fn set_transaction(value: &FfiSnapshotHintSetTransaction) -> DeltaResult<SetTransaction> {
    Ok(SetTransaction::new(
        unsafe { string(&value.app_id) }?,
        value.version,
        optional_i64(&value.last_updated),
    ))
}

unsafe fn domain_metadata(value: &FfiSnapshotHintDomainMetadata) -> DeltaResult<DomainMetadata> {
    let domain = unsafe { string(&value.domain) }?;
    let configuration = unsafe { string(&value.configuration) }?;
    Ok(if value.removed {
        DomainMetadata::remove(domain, configuration)
    } else {
        DomainMetadata::new(domain, configuration)
    })
}

unsafe fn checkpoint_metadata(
    value: &FfiSnapshotHintCheckpointMetadata,
) -> DeltaResult<CheckpointMetadata> {
    Ok(CheckpointMetadata::new(value.version, unsafe {
        optional_string_map(&value.tags)
    }?))
}

unsafe fn sidecar(value: &FfiSnapshotHintSidecar) -> DeltaResult<Sidecar> {
    Ok(Sidecar::new(
        unsafe { string(&value.path) }?,
        value.size_in_bytes,
        value.modification_time,
        unsafe { optional_string_map(&value.tags) }?,
    ))
}

unsafe fn action(value: &FfiSnapshotHintAction) -> DeltaResult<HintAction> {
    Ok(match value.kind {
        SNAPSHOT_HINT_ACTION_METADATA => HintAction::Metadata(unsafe {
            metadata(required_ref(value.value.metadata, "metadata action")?)?
        }),
        SNAPSHOT_HINT_ACTION_PROTOCOL => HintAction::Protocol(unsafe {
            protocol(required_ref(value.value.protocol, "protocol action")?)?
        }),
        SNAPSHOT_HINT_ACTION_TRANSACTION => HintAction::Txn(unsafe {
            set_transaction(required_ref(value.value.transaction, "transaction action")?)?
        }),
        SNAPSHOT_HINT_ACTION_DOMAIN_METADATA => HintAction::DomainMetadata(unsafe {
            domain_metadata(required_ref(
                value.value.domain_metadata,
                "domain-metadata action",
            )?)?
        }),
        SNAPSHOT_HINT_ACTION_CHECKPOINT_METADATA => HintAction::CheckpointMetadata(unsafe {
            checkpoint_metadata(required_ref(
                value.value.checkpoint_metadata,
                "checkpoint-metadata action",
            )?)?
        }),
        kind => {
            return Err(invalid(format!(
                "unknown snapshot hint action kind: {kind}"
            )))
        }
    })
}

unsafe fn v2_checkpoint(value: &FfiSnapshotHintV2Checkpoint) -> DeltaResult<LastCheckpointV2> {
    let sidecar_files = value
        .has_sidecar_files
        .then(|| unsafe {
            raw_slice(
                value.sidecar_files.ptr,
                value.sidecar_files.len,
                "sidecar array",
            )?
            .iter()
            .map(|value| sidecar(value))
            .collect()
        })
        .transpose()?;
    let non_file_actions = value
        .has_non_file_actions
        .then(|| unsafe {
            raw_slice(
                value.non_file_actions.ptr,
                value.non_file_actions.len,
                "non-file action array",
            )?
            .iter()
            .map(|value| action(value))
            .collect()
        })
        .transpose()?;
    Ok(LastCheckpointV2::from_parts(
        unsafe { string(&value.path) }?,
        optional_i64(&value.size_in_bytes),
        optional_i64(&value.modification_time),
        sidecar_files,
        non_file_actions,
    ))
}

unsafe fn last_checkpoint(
    value: &FfiSnapshotHintLastCheckpoint,
) -> DeltaResult<LastCheckpointHint> {
    let checkpoint_schema = unsafe { optional_string(&value.checkpoint_schema) }?;
    let v2_checkpoint = (!value.v2_checkpoint.is_null())
        .then(|| unsafe { v2_checkpoint(&*value.v2_checkpoint) })
        .transpose()?;
    LastCheckpointHint::from_parts(
        value.version,
        value.size,
        optional_usize(&value.parts)?,
        optional_i64(&value.size_in_bytes),
        optional_i64(&value.num_of_add_files),
        checkpoint_schema,
        unsafe { optional_string(&value.checksum) }?,
        unsafe { optional_string_map(&value.tags) }?,
        v2_checkpoint,
    )
}

unsafe fn i64s(value: &FfiI64Array) -> DeltaResult<Vec<i64>> {
    Ok(unsafe { raw_slice(value.ptr, value.len, "integer array") }?.to_vec())
}

unsafe fn crc(
    value: &FfiSnapshotHintCrc,
    version: Version,
    metadata: Metadata,
    protocol: Protocol,
) -> DeltaResult<Crc> {
    let file_size_histogram = (!value.file_size_histogram.is_null())
        .then(|| unsafe {
            let histogram = &*value.file_size_histogram;
            FileSizeHistogram::try_new(
                i64s(&histogram.sorted_bin_boundaries)?,
                i64s(&histogram.file_counts)?,
                i64s(&histogram.total_bytes)?,
            )
        })
        .transpose()?;
    let set_transactions = value
        .has_set_transactions
        .then(|| unsafe {
            raw_slice(
                value.set_transactions.ptr,
                value.set_transactions.len,
                "set-transaction array",
            )?
            .iter()
            .map(|value| set_transaction(value))
            .collect()
        })
        .transpose()?;
    let domain_metadata = value
        .has_domain_metadata
        .then(|| unsafe {
            raw_slice(
                value.domain_metadata.ptr,
                value.domain_metadata.len,
                "domain-metadata array",
            )?
            .iter()
            .map(|value| domain_metadata(value))
            .collect()
        })
        .transpose()?;
    Crc::try_new_complete(
        version,
        metadata,
        protocol,
        value.num_files,
        value.table_size_bytes,
        file_size_histogram,
        optional_i64(&value.in_commit_timestamp),
        set_transactions,
        domain_metadata,
    )
}

fn visitor(builder: &mut FfiSnapshotBuilder) -> DeltaResult<&mut SnapshotHintVisitorState> {
    match &mut builder.snapshot_hint {
        FfiSnapshotHintState::Building(visitor) => Ok(visitor),
        FfiSnapshotHintState::None | FfiSnapshotHintState::Ready(_) => {
            Err(invalid("snapshot hint visitor has not been started"))
        }
    }
}

fn report(builder: &FfiSnapshotBuilder, result: DeltaResult<bool>) -> ExternResult<bool> {
    unsafe { result.into_extern_result(&builder.engine.as_ref()) }
}

/// Begins typed snapshot-hint construction on a snapshot builder.
///
/// Calling this again discards any unfinished hint state on the builder.
/// `Latest` makes the built snapshot report `is_built_as_latest() == true`; kernel trusts this
/// caller claim. `Unverified` makes it report false.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when `version_status` is not a known value.
///
/// # Safety
///
/// `builder` must be a valid, exclusively borrowed snapshot-builder handle.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_begin(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    version: Version,
    version_status: FfiSnapshotHintVersionStatus,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = parse_version_status(version_status).map(|status| {
        builder.snapshot_hint =
            FfiSnapshotHintState::Building(Box::new(SnapshotHintVisitorState {
                version,
                version_status: status,
                log_paths: None,
                protocol: None,
                metadata: None,
                last_checkpoint_hint: None,
                crc: None,
            }));
        true
    });
    report(builder, result)
}

/// Copies the complete log-path set into an active snapshot-hint visitor.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active or a supplied path is invalid.
///
/// # Safety
///
/// The builder and every pointer reachable from `log_paths` must remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_set_log_paths(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    log_paths: LogPathArray,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = unsafe { log_paths.log_paths() }.and_then(|paths| {
        visitor(builder)?.log_paths = Some(paths);
        Ok(true)
    });
    report(builder, result)
}

/// Copies typed protocol state into an active snapshot-hint visitor.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active. Invalid protocol fields retain their
/// protocol-specific error code.
///
/// # Safety
///
/// The builder and every pointer reachable from `value` must remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_set_protocol(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    value: &FfiSnapshotHintProtocol,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = unsafe { protocol(value) }.and_then(|value| {
        visitor(builder)?.protocol = Some(value);
        Ok(true)
    });
    report(builder, result)
}

/// Copies typed metadata state into an active snapshot-hint visitor.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active. Invalid strings or schema fields retain
/// their source error code.
///
/// # Safety
///
/// The builder and every pointer reachable from `value` must remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_set_metadata(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    value: &FfiSnapshotHintMetadata,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = unsafe { metadata(value) }.and_then(|value| {
        visitor(builder)?.metadata = Some(value);
        Ok(true)
    });
    report(builder, result)
}

/// Copies typed `_last_checkpoint` state into an active snapshot-hint visitor.
///
/// Omit this call when the snapshot hint has no checkpoint hint.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active. Invalid checkpoint fields retain their
/// source error code.
///
/// # Safety
///
/// The builder and every pointer reachable from `value` must remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_set_last_checkpoint(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    value: &FfiSnapshotHintLastCheckpoint,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = unsafe { last_checkpoint(value) }.and_then(|value| {
        visitor(builder)?.last_checkpoint_hint = Some(value);
        Ok(true)
    });
    report(builder, result)
}

/// Copies typed CRC state into an active snapshot-hint visitor.
///
/// Protocol and metadata must be supplied before this call. Omit this call when the hint has no
/// CRC.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active, protocol or metadata is absent, or the
/// supplied CRC state is invalid.
///
/// # Safety
///
/// The builder and every pointer reachable from `value` must remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_set_crc(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
    value: &FfiSnapshotHintCrc,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let result = (|| {
        let state = visitor(builder)?;
        let metadata = state
            .metadata
            .clone()
            .ok_or_else(|| invalid("snapshot hint metadata must be supplied before CRC"))?;
        let protocol = state
            .protocol
            .clone()
            .ok_or_else(|| invalid("snapshot hint protocol must be supplied before CRC"))?;
        let crc =
            unsafe { crc(value, state.version, metadata, protocol) }.map_err(invalid_from_error)?;
        state.crc = Some(Arc::new(crc));
        Ok(true)
    })();
    report(builder, result)
}

/// Completes typed snapshot-hint construction and installs the hint on the snapshot builder.
///
/// Log paths, protocol, and metadata are required. Component parsing occurs in the setters;
/// cross-component and table validation occurs when the builder is built. This function consumes
/// the visitor even on error, so callers must call `snapshot_builder_snapshot_hint_begin` before
/// retrying.
///
/// # Errors
///
/// Returns `InvalidSnapshotHint` when no visitor is active or a required field is absent.
///
/// # Safety
///
/// `builder` must be a valid, exclusively borrowed snapshot-builder handle.
#[no_mangle]
pub unsafe extern "C" fn snapshot_builder_snapshot_hint_finish(
    builder: &mut Handle<MutableFfiSnapshotBuilder>,
) -> ExternResult<bool> {
    let builder = unsafe { builder.as_mut() };
    let state = std::mem::replace(&mut builder.snapshot_hint, FfiSnapshotHintState::None);
    let result = match state {
        FfiSnapshotHintState::Building(state) => Ok(state),
        FfiSnapshotHintState::None | FfiSnapshotHintState::Ready(_) => {
            Err(invalid("snapshot hint visitor has not been started"))
        }
    }
    .and_then(|state| {
        SnapshotHint::try_new(
            state.version,
            state
                .log_paths
                .ok_or_else(|| invalid("snapshot hint log paths were not supplied"))?,
            state
                .protocol
                .ok_or_else(|| invalid("snapshot hint protocol was not supplied"))?,
            state
                .metadata
                .ok_or_else(|| invalid("snapshot hint metadata was not supplied"))?,
            state.last_checkpoint_hint,
            state.crc,
            state.version_status,
        )
    })
    .map(|hint| {
        builder.snapshot_hint = FfiSnapshotHintState::Ready(Box::new(hint));
        true
    });
    report(builder, result)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use delta_kernel::object_store::memory::InMemory;
    use delta_kernel_default_engine::DefaultEngineBuilder;

    use super::*;
    use crate::error::KernelError;
    use crate::ffi_test_utils::{
        allocate_err, assert_extern_result_error_with_message, ok_or_panic,
    };
    use crate::log_path::FfiLogPath;
    use crate::{
        engine_to_handle, free_engine, free_snapshot, free_snapshot_builder, get_snapshot_builder,
        snapshot_builder_build,
    };

    fn slice(value: &'static str) -> KernelStringSlice {
        unsafe { KernelStringSlice::new_unsafe(value) }
    }

    fn none_string() -> FfiOptionalString {
        FfiOptionalString {
            has_value: false,
            value: slice(""),
        }
    }

    fn none_i64() -> FfiOptionalI64 {
        FfiOptionalI64 {
            has_value: false,
            value: 0,
        }
    }

    fn empty_strings(present: bool) -> FfiOptionalStringArray {
        FfiOptionalStringArray {
            has_value: present,
            value: FfiStringArray {
                ptr: std::ptr::null(),
                len: 0,
            },
        }
    }

    fn empty_map() -> FfiStringMap {
        FfiStringMap {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    fn none_map() -> FfiOptionalStringMap {
        FfiOptionalStringMap {
            has_value: false,
            value: empty_map(),
        }
    }

    fn test_protocol() -> FfiSnapshotHintProtocol {
        FfiSnapshotHintProtocol {
            min_reader_version: 1,
            min_writer_version: 2,
            reader_features: empty_strings(false),
            writer_features: empty_strings(false),
        }
    }

    fn test_metadata() -> FfiSnapshotHintMetadata {
        FfiSnapshotHintMetadata {
            id: slice("table-id"),
            name: none_string(),
            description: none_string(),
            format_provider: slice("parquet"),
            format_options: empty_map(),
            schema_string: slice(r#"{"type":"struct","fields":[]}"#),
            partition_columns: FfiStringArray {
                ptr: std::ptr::null(),
                len: 0,
            },
            created_time: none_i64(),
            configuration: empty_map(),
        }
    }

    #[test]
    fn typed_components_construct_rich_snapshot_state() {
        let protocol = unsafe { protocol(&test_protocol()) }.unwrap();
        let metadata = unsafe { metadata(&test_metadata()) }.unwrap();

        let transaction = FfiSnapshotHintSetTransaction {
            app_id: slice("app"),
            version: 7,
            last_updated: FfiOptionalI64 {
                has_value: true,
                value: 123,
            },
        };
        let domain = FfiSnapshotHintDomainMetadata {
            domain: slice("example.domain"),
            configuration: slice("payload"),
            removed: false,
        };
        let boundaries = [0, 1024];
        let counts = [1, 0];
        let bytes = [512, 0];
        let histogram = FfiSnapshotHintFileSizeHistogram {
            sorted_bin_boundaries: FfiI64Array {
                ptr: boundaries.as_ptr(),
                len: boundaries.len(),
            },
            file_counts: FfiI64Array {
                ptr: counts.as_ptr(),
                len: counts.len(),
            },
            total_bytes: FfiI64Array {
                ptr: bytes.as_ptr(),
                len: bytes.len(),
            },
        };
        let crc_value = FfiSnapshotHintCrc {
            table_size_bytes: 512,
            num_files: 1,
            in_commit_timestamp: none_i64(),
            file_size_histogram: &histogram,
            has_set_transactions: true,
            set_transactions: FfiSnapshotHintSetTransactionArray {
                ptr: &transaction,
                len: 1,
            },
            has_domain_metadata: true,
            domain_metadata: FfiSnapshotHintDomainMetadataArray {
                ptr: &domain,
                len: 1,
            },
        };
        let crc = unsafe { crc(&crc_value, 5, metadata.clone(), protocol.clone()) }.unwrap();
        assert_eq!(crc.file_stats().unwrap().num_files(), 1);
        assert_eq!(crc.set_transaction_state.expect_complete().len(), 1);
        assert_eq!(crc.domain_metadata_state.expect_complete().len(), 1);

        let checkpoint_metadata = FfiSnapshotHintCheckpointMetadata {
            version: 5,
            tags: none_map(),
        };
        let non_file_action = FfiSnapshotHintAction {
            kind: SNAPSHOT_HINT_ACTION_CHECKPOINT_METADATA,
            value: FfiSnapshotHintActionValue {
                checkpoint_metadata: &checkpoint_metadata,
            },
        };
        let sidecar = FfiSnapshotHintSidecar {
            path: slice("sidecar.parquet"),
            size_in_bytes: 42,
            modification_time: 123,
            tags: none_map(),
        };
        let v2 = FfiSnapshotHintV2Checkpoint {
            path: slice("00000000000000000005.checkpoint.uuid.parquet"),
            size_in_bytes: none_i64(),
            modification_time: none_i64(),
            has_sidecar_files: true,
            sidecar_files: FfiSnapshotHintSidecarArray {
                ptr: &sidecar,
                len: 1,
            },
            has_non_file_actions: true,
            non_file_actions: FfiSnapshotHintActionArray {
                ptr: &non_file_action,
                len: 1,
            },
        };
        let checkpoint = FfiSnapshotHintLastCheckpoint {
            version: 5,
            size: 1,
            parts: FfiOptionalU64 {
                has_value: false,
                value: 0,
            },
            size_in_bytes: none_i64(),
            num_of_add_files: none_i64(),
            checkpoint_schema: none_string(),
            checksum: none_string(),
            tags: none_map(),
            v2_checkpoint: &v2,
        };
        let checkpoint = unsafe { last_checkpoint(&checkpoint) }.unwrap();
        assert_eq!(checkpoint.version, 5);
    }

    #[test]
    fn typed_arrays_preserve_absent_and_present_empty() {
        assert_eq!(
            unsafe { optional_strings(&empty_strings(false)) }.unwrap(),
            None
        );
        assert_eq!(
            unsafe { optional_strings(&empty_strings(true)) }.unwrap(),
            Some(vec![])
        );
    }

    #[test]
    fn typed_array_rejects_null_nonempty_pointer() {
        let invalid = FfiStringArray {
            ptr: std::ptr::null(),
            len: 1,
        };
        assert!(unsafe { strings(&invalid) }.is_err());
    }

    #[test]
    fn typed_actions_validate_tags_and_convert_each_payload() {
        let metadata = test_metadata();
        let protocol = test_protocol();
        let transaction = FfiSnapshotHintSetTransaction {
            app_id: slice("app"),
            version: 1,
            last_updated: none_i64(),
        };
        let domain_metadata = FfiSnapshotHintDomainMetadata {
            domain: slice("domain"),
            configuration: slice("{}"),
            removed: false,
        };
        let checkpoint_metadata = FfiSnapshotHintCheckpointMetadata {
            version: 1,
            tags: none_map(),
        };
        let actions = [
            FfiSnapshotHintAction {
                kind: SNAPSHOT_HINT_ACTION_METADATA,
                value: FfiSnapshotHintActionValue {
                    metadata: &metadata,
                },
            },
            FfiSnapshotHintAction {
                kind: SNAPSHOT_HINT_ACTION_PROTOCOL,
                value: FfiSnapshotHintActionValue {
                    protocol: &protocol,
                },
            },
            FfiSnapshotHintAction {
                kind: SNAPSHOT_HINT_ACTION_TRANSACTION,
                value: FfiSnapshotHintActionValue {
                    transaction: &transaction,
                },
            },
            FfiSnapshotHintAction {
                kind: SNAPSHOT_HINT_ACTION_DOMAIN_METADATA,
                value: FfiSnapshotHintActionValue {
                    domain_metadata: &domain_metadata,
                },
            },
            FfiSnapshotHintAction {
                kind: SNAPSHOT_HINT_ACTION_CHECKPOINT_METADATA,
                value: FfiSnapshotHintActionValue {
                    checkpoint_metadata: &checkpoint_metadata,
                },
            },
        ];
        for value in &actions {
            unsafe { action(value) }.unwrap();
        }

        let invalid_action = FfiSnapshotHintAction {
            kind: u32::MAX,
            value: FfiSnapshotHintActionValue {
                metadata: std::ptr::null(),
            },
        };
        assert!(matches!(
            unsafe { action(&invalid_action) },
            Err(Error::InvalidSnapshotHint(_))
        ));
        let null_action = FfiSnapshotHintAction {
            kind: SNAPSHOT_HINT_ACTION_METADATA,
            value: FfiSnapshotHintActionValue {
                metadata: std::ptr::null(),
            },
        };
        assert!(matches!(
            unsafe { action(&null_action) },
            Err(Error::InvalidSnapshotHint(_))
        ));
    }

    #[test]
    fn typed_visitor_rejects_unknown_freshness_and_unfinished_build() {
        let engine = engine_to_handle(
            Arc::new(DefaultEngineBuilder::new(Arc::new(InMemory::new())).build()),
            allocate_err,
        );
        let mut builder = unsafe {
            ok_or_panic(get_snapshot_builder(
                slice("memory:///hinted-table/"),
                engine.shallow_copy(),
            ))
        };
        let result = unsafe { snapshot_builder_snapshot_hint_begin(&mut builder, 0, u32::MAX) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some("Invalid snapshot hint: unknown snapshot hint version status: 4294967295"),
        );
        let result =
            unsafe { snapshot_builder_snapshot_hint_set_metadata(&mut builder, &test_metadata()) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some("Invalid snapshot hint: snapshot hint visitor has not been started"),
        );

        unsafe {
            ok_or_panic(snapshot_builder_snapshot_hint_begin(
                &mut builder,
                0,
                SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED,
            ));
        }
        let result = unsafe { snapshot_builder_build(builder) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some("Invalid snapshot hint: snapshot hint visitor is unfinished"),
        );
        unsafe { free_engine(engine) };
    }

    #[test]
    fn typed_crc_reports_malformed_histogram_as_invalid_snapshot_hint() {
        let engine = engine_to_handle(
            Arc::new(DefaultEngineBuilder::new(Arc::new(InMemory::new())).build()),
            allocate_err,
        );
        let mut builder = unsafe {
            ok_or_panic(get_snapshot_builder(
                slice("memory:///hinted-table/"),
                engine.shallow_copy(),
            ))
        };
        unsafe {
            ok_or_panic(snapshot_builder_snapshot_hint_begin(
                &mut builder,
                0,
                SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED,
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_set_protocol(
                &mut builder,
                &test_protocol(),
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_set_metadata(
                &mut builder,
                &test_metadata(),
            ));
        }

        let boundary = [0];
        let histogram = FfiSnapshotHintFileSizeHistogram {
            sorted_bin_boundaries: FfiI64Array {
                ptr: boundary.as_ptr(),
                len: boundary.len(),
            },
            file_counts: FfiI64Array {
                ptr: boundary.as_ptr(),
                len: boundary.len(),
            },
            total_bytes: FfiI64Array {
                ptr: boundary.as_ptr(),
                len: boundary.len(),
            },
        };
        let crc = FfiSnapshotHintCrc {
            table_size_bytes: 0,
            num_files: 0,
            in_commit_timestamp: none_i64(),
            file_size_histogram: &histogram,
            has_set_transactions: false,
            set_transactions: FfiSnapshotHintSetTransactionArray {
                ptr: std::ptr::null(),
                len: 0,
            },
            has_domain_metadata: false,
            domain_metadata: FfiSnapshotHintDomainMetadataArray {
                ptr: std::ptr::null(),
                len: 0,
            },
        };
        let result = unsafe { snapshot_builder_snapshot_hint_set_crc(&mut builder, &crc) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some(
                "Invalid snapshot hint: sorted_bin_boundaries must have at least 2 elements, got 1",
            ),
        );

        unsafe {
            free_snapshot_builder(builder);
            free_engine(engine);
        }
    }

    #[test]
    fn typed_visitor_builds_latest_snapshot_without_storage_files() {
        let engine = engine_to_handle(
            Arc::new(DefaultEngineBuilder::new(Arc::new(InMemory::new())).build()),
            allocate_err,
        );
        let mut builder = unsafe {
            ok_or_panic(get_snapshot_builder(
                slice("memory:///hinted-table/"),
                engine.shallow_copy(),
            ))
        };
        let log_path = FfiLogPath::new(
            slice("memory:///hinted-table/_delta_log/00000000000000000000.json"),
            1,
            1,
        );

        unsafe {
            ok_or_panic(snapshot_builder_snapshot_hint_begin(
                &mut builder,
                0,
                SNAPSHOT_HINT_VERSION_STATUS_LATEST,
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_set_log_paths(
                &mut builder,
                LogPathArray {
                    ptr: &log_path,
                    len: 1,
                },
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_set_protocol(
                &mut builder,
                &test_protocol(),
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_set_metadata(
                &mut builder,
                &test_metadata(),
            ));
            ok_or_panic(snapshot_builder_snapshot_hint_finish(&mut builder));
        }

        let snapshot = unsafe { ok_or_panic(snapshot_builder_build(builder)) };
        let snapshot_ref = unsafe { snapshot.as_ref() };
        assert_eq!(snapshot_ref.version(), 0);
        assert!(snapshot_ref.is_built_as_latest());

        unsafe {
            free_snapshot(snapshot);
            free_engine(engine);
        }
    }

    #[test]
    fn typed_visitor_rejects_missing_fields_and_partial_state_can_be_freed() {
        let engine = engine_to_handle(
            Arc::new(DefaultEngineBuilder::new(Arc::new(InMemory::new())).build()),
            allocate_err,
        );
        let mut missing_fields_builder = unsafe {
            ok_or_panic(get_snapshot_builder(
                slice("memory:///hinted-table/"),
                engine.shallow_copy(),
            ))
        };
        unsafe {
            ok_or_panic(snapshot_builder_snapshot_hint_begin(
                &mut missing_fields_builder,
                0,
                SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED,
            ));
        }
        let result = unsafe { snapshot_builder_snapshot_hint_finish(&mut missing_fields_builder) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some("Invalid snapshot hint: snapshot hint log paths were not supplied"),
        );
        let result = unsafe { snapshot_builder_snapshot_hint_finish(&mut missing_fields_builder) };
        assert_extern_result_error_with_message(
            result,
            KernelError::InvalidSnapshotHint,
            Some("Invalid snapshot hint: snapshot hint visitor has not been started"),
        );

        let mut partial_builder = unsafe {
            ok_or_panic(get_snapshot_builder(
                slice("memory:///hinted-table/"),
                engine.shallow_copy(),
            ))
        };
        unsafe {
            ok_or_panic(snapshot_builder_snapshot_hint_begin(
                &mut partial_builder,
                0,
                SNAPSHOT_HINT_VERSION_STATUS_UNVERIFIED,
            ));
            free_snapshot_builder(missing_fields_builder);
            free_snapshot_builder(partial_builder);
            free_engine(engine);
        }
    }
}
