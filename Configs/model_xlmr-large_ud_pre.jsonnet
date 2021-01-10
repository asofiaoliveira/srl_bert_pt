{
    "dataset_reader":{
        "type":"ud_transformer",
        "bert_model_name": "xlm-roberta-large",
        "token_indexers": {
            "tokens": {
              "type": "pretrained_transformer",
            	"model_name": "xlm-roberta-large"
            }
        }
    },
    
    "model": {
      "type": "biaffine_parser",
      "encoder": {
            "type": "pass_through",
            "input_dim": 1024
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                  "type": "pretrained_transformer_mine",
                  "model_name": "xlm-roberta-large"
                }
        }
      },
      
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 4
      }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}]
            ]
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular"
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "grad_norm": 1.0,
        "num_epochs": 10,
        "validation_metric": "+LAS",
        "cuda_device": 0
    },

    "vocabulary": {
        "type": "from_instances"
    }
}
