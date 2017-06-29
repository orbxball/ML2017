import numpy as np
import pandas as pd

class DataProcessor():

    def __init__(self):
        pass

    def read_data(self, train_f, test_f, label_f):
        self.raw_train_features = pd.read_csv(train_f)
        self.raw_test_features = pd.read_csv(test_f)
        self.raw_labels = pd.read_csv(label_f)

    def preprocess(self):
        train = self.raw_train_features
        test = self.raw_test_features
        labels = self.raw_labels

        train['test'] = 0
        test['test'] = 1

        data = pd.concat([train, test])

        data['date_recorded'] = pd.to_datetime(data['date_recorded'])
        data['date_recorded'] = (data['date_recorded'] - data['date_recorded'].min()) / np.timedelta64(1, 'D')

        data['construction_year'] = data['construction_year'] - 1960
        data['construction_year'][data['construction_year'] < 0] = data['construction_year'][data['construction_year'] >= 0].median()

        data['gps_height'][data['gps_height'] == 0] = data['gps_height'][data['gps_height'] > 0].median()

        data.drop(['num_private', 'recorded_by', 'wpt_name', 'extraction_type_group', 'extraction_type', 'payment_type',
            'water_quality', 'scheme_management', 'district_code', 'region', 'region_code', 'subvillage', 'ward',
            'waterpoint_type_group', 'quantity_group', 'installer'], axis=1, inplace=True)

        columns = list(data.select_dtypes(include=['object']).columns)

        data = pd.get_dummies(data, columns=columns)

        train = data.loc[data['test'] == 0]
        test = data.loc[data['test'] == 1]

        self.id = test['id']

        train.drop(['id', 'test'], axis=1, inplace=True)
        test.drop(['id', 'test'], axis=1, inplace=True)

        labels.drop(['id'], axis=1, inplace=True)
        labels = labels['status_group'].astype('category')

        labels.cat.reorder_categories(['non functional', 'functional needs repair', 'functional'], inplace=True)

        self.train = train.values
        self.test = test.values
        self.labels = labels.cat.codes.values
        self.classes = labels.cat.categories

    def write_data(self, filename, pred): 
        pred = [self.classes[i] for i in pred]

        output = pd.DataFrame({'id': self.id, 'status_group': pred}, columns=['id', 'status_group'])
        output.to_csv(filename, index=False, columns=('id', 'status_group'))
