"""Public, actionable domain errors."""


class PPGError(RuntimeError):
    code = "PPG_ERROR"


class DataAlignmentError(PPGError):
    code = "DATA_ALIGNMENT_ERROR"


class DataValidationError(PPGError):
    code = "DATA_VALIDATION_ERROR"


class DataBlockedError(PPGError):
    code = "DATA_BLOCKED"


class SplitLeakageError(PPGError):
    code = "SPLIT_LEAKAGE_ERROR"


class PredictionIntegrityError(PPGError):
    code = "PREDICTION_INTEGRITY_ERROR"


class PipelineMismatchError(PPGError):
    code = "PIPELINE_MISMATCH_ERROR"


class ConfigurationError(PPGError):
    code = "CONFIGURATION_ERROR"
