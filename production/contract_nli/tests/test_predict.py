'''
Here is where we want to test inputs, outputs and model quality
below are sample tests 
'''

#Testing for entailment condition
def test_prediction_quality_against_benchmark_entailment(load_predict_class, sample_input_data_1):
    res = load_predict_class.make_single_prediction(sample_input_data_1["premise"].split(), sample_input_data_1["hypothesis"].split())
    assert res == sample_input_data_1["result"]

#Testing for Contradiction condition
def test_prediction_quality_against_benchmark_entailment(load_predict_class, sample_input_data_2):
    res = load_predict_class.make_single_prediction(sample_input_data_2["premise"].split(), sample_input_data_2["hypothesis"].split())
    assert res == sample_input_data_2["result"]

#Testing for neutral condition
def test_prediction_quality_against_benchmark_entailment(load_predict_class, sample_input_data_3):
    res = load_predict_class.make_single_prediction(sample_input_data_3["premise"].split(), sample_input_data_3["hypothesis"].split())
    assert res == sample_input_data_3["result"]