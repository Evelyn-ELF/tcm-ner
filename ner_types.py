"""实体抽取的结构化接口；仅描述数据，不替代运行时输入校验。"""

from typing import NotRequired, TypedDict


class Entity(TypedDict):
    """单个实体，start/end 为原文 Python 字符索引的左闭右开区间。"""

    text: str
    type: str
    start: int
    end: int


class InputSample(TypedDict):
    """推理输入；instruction 缺省时使用项目固定模板。"""

    id: str
    input: str
    instruction: NotRequired[str]


class Sample(InputSample):
    """带真实实体标注的训练或评估样本，空标注使用空列表。"""

    output: list[Entity]


class ItemError(TypedDict):
    """生成数组中非法条目的序号和原因。"""

    index: int
    reason: str


class ParsedPrediction(TypedDict):
    """解析结果；非法 JSON 通过 parse_error 返回，不自动修补。"""

    entities: list[Entity]
    parse_error: str | None
    invalid_by_type: dict[str, int]
    unknown_type_fp: int
    errors: list[ItemError]


class Prediction(InputSample):
    """生成记录；只有评估样本携带 output 真实标注。"""

    output: NotRequired[list[Entity]]
    raw_output: str
    predicted_entities: list[Entity]
    parse_error: str | None
    item_errors: list[ItemError]
    hit_generation_limit: bool


class EncodedSample(TypedDict):
    """等长的 token、注意力掩码和标签，非监督位置的标签为 -100。"""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class EntityMetrics(TypedDict):
    """实体级统计，分母为零时 precision、recall 和 f1 均按零处理。"""

    precision: float
    recall: float
    f1: float
    support: int
    tp: int
    fp: int
    fn: int


class MetricsReport(TypedDict):
    """完整计分报告，macro_f1 固定对全部实体类别平均。"""

    sample_count: int
    per_type: dict[str, EntityMetrics]
    micro: EntityMetrics
    macro_f1: float
    parse_failure_rate: float
    parse_failure_count: int
    invalid_item_count: int
    unknown_type_fp: int
    generation_limit_rate: float
    scoring_rule: str
