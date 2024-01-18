from src.Anomaly_calculation.anomaly_calculation import ExtremeAnoMap
import pickle

if __name__=='__main__':
    field_num=213
    with open(f'../../output/Pickled_files/Field_data/Field_{field_num}_data.pickle', 'rb') as f:
        field_dict=pickle.load(f)

    with open(f'../../output/Pickled_files/PDFs/Field_{field_num}_original_doy.pkl', 'rb') as file:
        propa=pickle.load(file)
    
    one_field=ExtremeAnoMap(propa, field_dict[field_num]['original_doy_ts'], field_dict[field_num]['NDVI_ts'])

    rfd=one_field.rfd()
    delta=one_field.delta()
    print('Finish')