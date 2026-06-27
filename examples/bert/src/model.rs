use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::LayerNorm;

//bert-base-uncased hyperparameters to use as default
pub const LAYERS: usize = 12;
pub const HIDDEN: usize = 768;
pub const INTERMEDIATE: usize = 3072;
pub const HEAD_DIM: usize = 64;
pub const NUM_HEADS: usize = 12;
pub const MAX_POSITION_EMBEDDINGS: usize = 512;
pub const VOCAB_SIZE: usize = 30522;
pub const TYPE_VOCAB_SIZE: usize = 2;
pub const LAYER_NORM_EPS: f32 = 1e-12;

//BertConfig for bert-base-uncased 
#[derive(Debug, Clone, Copy)]
pub struct BertConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
}

/// Configuration for BERT models. Defaults to bert-base-uncased hyperparams.
/// Pass a custom BertConfig to support other BERT variants (e.g. bert-large-uncased).
impl Default for BertConfig {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            num_heads: NUM_HEADS,
            vocab_size: VOCAB_SIZE,
            max_position_embeddings: MAX_POSITION_EMBEDDINGS,
            type_vocab_size: TYPE_VOCAB_SIZE,
        }
    }
}

impl BertConfig {
    pub fn head_dim(self) -> usize {
        self.hidden / self.num_heads // 64
    }
}

pub struct BertEmbeddings {
    pub word_embeddings: GraphTensor,
    pub position_embeddings: GraphTensor,
    pub token_type_embeddings: GraphTensor,
    pub embeddings_layer_norm: LayerNorm,
}

impl BertEmbeddings {
    pub fn init(cx: &mut Graph, config: BertConfig) -> Self {
        Self {
            word_embeddings: persist(cx, "embeddings.word_embeddings.weight", (config.vocab_size, config.hidden)),
            position_embeddings: persist(cx, "embeddings.position_embeddings.weight", (config.max_position_embeddings, config.hidden)),
            token_type_embeddings: persist(cx, "embeddings.token_type_embeddings.weight", (config.type_vocab_size, config.hidden)),
            embeddings_layer_norm: LayerNorm::new(
                config.hidden,
                Some("embeddings.LayerNorm.weight"),
                Some("embeddings.LayerNorm.bias"),
                true,   // mean_norm: true = full LayerNorm, not RMSNorm
                LAYER_NORM_EPS,
                cx,
            ),
        }
    }

    pub fn forward(
        &self,
        input_ids: GraphTensor,
        position_ids: GraphTensor,
        token_type_ids: GraphTensor,
        config: BertConfig,) -> GraphTensor 
    {
        let hidden = config.hidden;
        let words = self.word_embeddings.gather(
            (input_ids * hidden).expand_dim(1, hidden)
                + input_ids.graph().arange(hidden).expand_dim(0, input_ids.dims1()),
        );
        let positions = self.position_embeddings.gather(
            (position_ids * hidden).expand_dim(1, hidden)
                + position_ids.graph().arange(hidden).expand_dim(0, input_ids.dims1()),
        );
        let token_types = self.token_type_embeddings.gather(
            (token_type_ids * hidden).expand_dim(1, hidden)
                + token_type_ids.graph().arange(hidden).expand_dim(0, input_ids.dims1()),
        );
        // sum all three embeddings and normalize
        self.embeddings_layer_norm.forward(words + positions + token_types)
    }
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

pub struct BertAttention {

    pub query_weight: GraphTensor,
    pub query_bias: GraphTensor,
    pub key_weight: GraphTensor,
    pub key_bias: GraphTensor,
    pub value_weight: GraphTensor,
    pub value_bias: GraphTensor,
    pub output_dense_weight: GraphTensor,
    pub output_dense_bias: GraphTensor,
    pub layer_norm: LayerNorm,
}

impl BertAttention {
    pub fn init(cx: &mut Graph, config: BertConfig, layer: usize) -> Self {
        Self {
            query_weight: persist(cx, format!("encoder.layer.{}.attention.self.query.weight", layer), (config.hidden, config.hidden)),
            query_bias: persist(cx, format!("encoder.layer.{}.attention.self.query.bias", layer), config.hidden),
            key_weight: persist(cx, format!("encoder.layer.{}.attention.self.key.weight", layer), (config.hidden, config.hidden)),
            key_bias: persist(cx, format!("encoder.layer.{}.attention.self.key.bias", layer), config.hidden),
            value_weight: persist(cx, format!("encoder.layer.{}.attention.self.value.weight", layer), (config.hidden, config.hidden)),
            value_bias: persist(cx, format!("encoder.layer.{}.attention.self.value.bias", layer), config.hidden),
            output_dense_weight: persist(cx, format!("encoder.layer.{}.attention.output.dense.weight", layer), (config.hidden, config.hidden)),
            output_dense_bias: persist(cx, format!("encoder.layer.{}.attention.output.dense.bias", layer), config.hidden),
            layer_norm: LayerNorm::new(
                config.hidden,
                Some(&format!("encoder.layer.{layer}.attention.output.LayerNorm.weight")),
                Some(&format!("encoder.layer.{layer}.attention.output.LayerNorm.bias")),
                true,
                LAYER_NORM_EPS,
                cx,
            )
        }
    }

    pub fn forward(&self, x:GraphTensor , config: BertConfig) -> GraphTensor
    {
        //projections
        let leading = &x.dims()[..x.dims().len() - 1];
        let q = x.matmul(self.query_weight.t())
        + self.query_bias.expand_lhs(leading);
        let k = x.matmul(self.key_weight.t())
        + self.key_bias.expand_lhs(leading);
        let v = x.matmul(self.value_weight.t())
        + self.value_bias.expand_lhs(leading);

        //head splitting
        let q = q.split_dims(1, config.head_dim()).transpose(0, 1);
        let k = k.split_dims(1, config.head_dim()).transpose(0, 1);
        let v = v.split_dims(1, config.head_dim()).transpose(0, 1);

        let scores = q.matmul(k.transpose(1,2))/8.0;  // Q @ K.T, scaled
        let weights = scores.softmax(2);// softmax
        let attn_out = weights.matmul(v);// weights @ V

        // merge heads back: (12, seq, 64) → (seq, 768)
        let attn_out = attn_out.transpose(0, 1).merge_dims(1, 2);

        let out = attn_out.matmul(self.output_dense_weight.t()) + self.output_dense_bias.expand_lhs(leading);
        self.layer_norm.forward(x + out)
    }

}

pub struct BertIntermediate
{
    pub weight: GraphTensor,
    pub bias: GraphTensor,
}

impl BertIntermediate
{
    pub fn init(cx: &mut Graph, config: BertConfig, i: usize) -> Self
    {
        Self
        {
            weight: persist(cx, format!("encoder.layer.{i}.intermediate.dense.weight"), (config.intermediate, config.hidden)),
            bias: persist(cx, format!("encoder.layer.{i}.intermediate.dense.bias"), config.intermediate),
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor
    {
        let leading = &x.dims()[..x.dims().len() - 1];
        let result = x.matmul(self.weight.t()) + self.bias.expand_lhs(leading);
        result.gelu()
    }
}

pub struct BertOutput
{
    pub output_dense_weight: GraphTensor,
    pub output_dense_bias: GraphTensor,
    pub layer_norm: LayerNorm,
}

impl BertOutput
{
    pub fn init(cx :&mut Graph, config: BertConfig, layer: usize) -> Self
    {
        Self
        {
            output_dense_weight: persist(cx, format!("encoder.layer.{layer}.output.dense.weight"), (config.hidden, config.intermediate)),
            output_dense_bias: persist(cx, format!("encoder.layer.{layer}.output.dense.bias"),(config.hidden)),
            layer_norm: LayerNorm::new(
                config.hidden,
                Some(&format!("encoder.layer.{layer}.output.LayerNorm.weight")),
                Some(&format!("encoder.layer.{layer}.output.LayerNorm.bias")),
                true,
                LAYER_NORM_EPS,
                cx,
            ),
        }
    }

    pub fn forward(&self, intermediate: GraphTensor, residual: GraphTensor) -> GraphTensor {
        let leading = &intermediate.dims()[..intermediate.dims().len() - 1];
        let projected = intermediate.matmul(self.output_dense_weight.t()) + self.output_dense_bias.expand_lhs(leading);
        self.layer_norm.forward(residual + projected)
    }
}

pub struct BertLayer
{
    pub attention: BertAttention,
    pub intermediate: BertIntermediate,
    pub output: BertOutput,
}

impl BertLayer
{
    pub fn init(cx :&mut Graph, config: BertConfig, layer: usize)-> Self
    {
        Self{
            attention : BertAttention::init(cx, config, layer),
            intermediate : BertIntermediate::init(cx, config, layer),
            output : BertOutput::init(cx, config, layer),
        }
        
    }

    pub fn forward(&self, x: GraphTensor, config: BertConfig) -> GraphTensor
    {
        let x = self.attention.forward(x, config);
        let intermediate = self.intermediate.forward(x);
        self.output.forward(intermediate, x)
    }
}

pub struct Bert
{
    pub embeddings: BertEmbeddings,
    pub layers: Vec<BertLayer>,
    pub pooler: BertPooler,
}

impl Bert
{
    pub fn init(cx :&mut Graph, config: BertConfig)->Self
    {
        let mut layers = Vec::with_capacity(12);
            for i in 0..config.layers{
                layers.push(BertLayer::init(cx, config, i));
            }
        Self{
            embeddings: BertEmbeddings::init(cx, config),
            pooler: BertPooler::init(cx, config),
            layers,
        }
    }

    pub fn forward(&self, input_ids: GraphTensor, position_ids: GraphTensor, token_type_ids: GraphTensor, config: BertConfig) ->(GraphTensor, GraphTensor)
    {
        let mut x = self.embeddings.forward(input_ids, position_ids, token_type_ids, config);
        for layer in &self.layers
        {
            x = layer.forward(x, config);
        }
        let pooled = self.pooler.forward(x);
        (x, pooled)
    }

}

pub struct BertPooler
{
    pub dense_weight: GraphTensor,
    pub dense_bias: GraphTensor,
}

impl BertPooler
{
    pub fn init(cx :&mut Graph, config: BertConfig) ->Self
    {
        Self
        {
            dense_weight: persist(cx, "pooler.dense.weight", (config.hidden, config.hidden)),
            dense_bias: persist(cx, "pooler.dense.bias", config.hidden)
        }
    }

    pub fn forward(&self, hidden_states: GraphTensor) -> GraphTensor {
        // [CLS] is token 0 → first row of (seq, hidden)
        let cls = hidden_states.slice((0..1, ..));
    
        let leading = &cls.dims()[..cls.dims().len() - 1];
        let pooled = cls.matmul(self.dense_weight.t()) + self.dense_bias.expand_lhs(leading);
        pooled.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::graph::Graph;

    #[test]
    fn test_bert_init_builds_graph() {
        let mut cx = Graph::new();
        let config = BertConfig::default();
        let _bert = Bert::init(&mut cx, config);
        // if this doesn't panic, graph construction is correct
    }

    #[test]
    fn test_bert_config_head_dim() {
        let config = BertConfig::default();
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_bert_config_defaults() {
        let config = BertConfig::default();
        assert_eq!(config.layers, 12);
        assert_eq!(config.hidden, 768);
        assert_eq!(config.intermediate, 3072);
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.max_position_embeddings, 512);
    }
}