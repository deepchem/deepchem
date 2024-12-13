import os
import pytest

try:
    import torch
    from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
except ModuleNotFoundError:
    pass


@pytest.fixture
def igbert_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Exscientia/IgBert', do_lower_case=False)
    return tokenizer


# @pytest.mark.hf
# def test_init(igbert_tokenizer):
#     from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
#     from deepchem.models.torch_models.hf_models import HuggingFaceModel
#     anti_model = DeepAbLLM(task='mlm', model_path='Exscientia/IgBert')
#     assert isinstance(anti_model, HuggingFaceModel)
#     assert anti_model.tokenizer == igbert_tokenizer
#     assert anti_model.n_tasks == 1


@pytest.mark.hf
def test_load_from_pretrained(tmpdir):
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    model_path = 'Rostlab/prot_bert'
    pretrain_model = DeepAbLLM(task='mlm',
                               model_path=model_path,
                               n_tasks=1,
                               model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = DeepAbLLM(task='mlm',
                               model_path=model_path,
                               n_tasks=1,
                               model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]
    assert all(matches)


@pytest.mark.hf
def test_initialize_new_config():
    model_path = 'Rostlab/prot_bert'
    config = {"num_attention_heads": 8, "num_hidden_layers": 6}
    model = DeepAbLLM(
        task='mlm',
        model_path=model_path,
        n_tasks=1,
        config=config,
    )

    assert model.model.config.num_attention_heads == 8
    assert model.model.config.num_hidden_layers == 6


@pytest.mark.hf
def test_save_reload(tmpdir):
    model_path = 'Exscientia/IgBert'
    anti_model = DeepAbLLM(task='mlm',
                           model_path=model_path,
                           n_tasks=1,
                           model_dir=tmpdir)
    anti_model._ensure_built()
    anti_model.save_checkpoint()

    anti_model2 = DeepAbLLM(task='mlm',
                            model_path=model_path,
                            n_tasks=1,
                            model_dir=tmpdir)
    anti_model2.restore()

    old_state = anti_model.model.state_dict()
    new_state = anti_model2.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)


@pytest.mark.hf
def test_mask_seq_pos(igbert_tokenizer):
    from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
    anti_model = DeepAbLLM(model_path='facebook/esm2_t6_8M_UR50D',
                           task='mlm',
                           is_esm_variant=True,
                           device=torch.device('cpu'))
    anti_model._ensure_built()

    test_string = "VQLAQSGSELRKPGASVKVSCDTSGHSFTSNAIHWVRQAPGQGLEWMGWINTDTGTPTYAQGFTGRFVFSLDTSARTAYLQISSLKADDTAVFYCARERDYSDYFFDYWGQGTLVTVSS"
    masked_index = 10
    masked_test_string = anti_model._mask_seq_pos(test_string, idx=masked_index)
    assert isinstance(masked_test_string, str)
    assert masked_test_string.index('<mask>') == masked_index


@pytest.mark.hf
def test_redesign_residue():
    from deepchem.utils import levenshtein_distance
    from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
    anti_model = DeepAbLLM(model_path='Rostlab/prot_bert', task='mlm')
    anti_model._ensure_built()
    ab_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSNYAMYWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRTEDTAVYYCASGSDYGDYLLVYWGQGTLVTVSS"
    hypotheses = anti_model.redesign_residue(ab_sequence,
                                             residue_index=0,
                                             top_k=8)
    assert len(hypotheses) == 8
    for item in hypotheses:
        # Assert that the tuples are of (token_str, full_seq, score)
        assert len(item) == 3
        # Test that the first item is a string
        assert isinstance(item[0], str)
        # assert len(item[0]) == 1
        # Test that the second item is a string
        assert len(item[1]) == len(ab_sequence)
        assert levenshtein_distance(item[1], ab_sequence) <= 1
        # Test the third item is a float between 0 and 1
        assert isinstance(item[2], float)
        assert abs(item[2]) <= 1


@pytest.mark.hf
def test_optimize_sequence():
    from deepchem.utils import levenshtein_distance
    from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
    anti_model = DeepAbLLM(model_path='Exscientia/IgBert', task='mlm')
    anti_model._ensure_built()
    ab_sequence = "ALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVSNRFSGSKSGNTASLTISGLQSEDEADYYCNSLTSISTWVFGGGTKLTVL"
    redesigned_sequences = anti_model.redesign_sequence(ab_sequence)
    assert len(redesigned_sequences) > 0
    for item in redesigned_sequences:
        # Assert that the tuples are of (index, token_str, full_seq, score)
        assert len(item) == 4
        # Test that the first item is a string
        assert isinstance(item[1], str)
        # assert len(item[1]) == 1
        # Test that the second item is a string
        assert len(item[2]) == len(ab_sequence)
        assert distance(item[2], ab_sequence) <= 1
        # Test the third item is a float between 0 and 1
        assert isinstance(item[3], float)
        assert abs(item[3]) <= 1
