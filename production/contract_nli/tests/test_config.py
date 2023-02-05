'''
We do not want to change these as they are static aspects 
of this solution. If they are changed we want fail tests.
'''
def test_app_config(raw_app_config):
    assert raw_app_config.package_name == "Contract_NLI"
    assert raw_app_config.test_path == 'test.csv'
    assert raw_app_config.vocab_path== "vocab.pth"
    assert raw_app_config.model_path == "model.pth"

'''
These are model training properties we dont want to change, if they are changed
we need to update tests. 
'''
def test_model_config(raw_model_config):
    assert raw_model_config.batch_size == 256
    assert raw_model_config.embed_size == 100
    assert raw_model_config.learning_rate == 0.01
    assert raw_model_config.trainer == "Adam"
