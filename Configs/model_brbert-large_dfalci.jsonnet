{

    "dataset_reader": {
        "type": "srl-pt",
	    "bert_model_name": "neuralmind/bert-large-portuguese-cased",
        "token_indexers": {
            "tokens": {
              "type": "pretrained_transformer",
            	"model_name": "neuralmind/bert-large-portuguese-cased"
            }
        }
      },
    
    "evaluate_on_test": true,
    
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 16
      }
    },
    "model": {
        "type": "my_model",
        "embedding_dropout": 0.1,
        "verbose_metrics": true,
        "bert_model": "neuralmind/bert-large-portuguese-cased"
    },

    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 4e-5,
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
        "num_epochs": 100,
        "patience": 10,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0
    },

    "vocabulary": {
        "type": "from_instances"
    }
}
