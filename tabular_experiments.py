import numpy as np
from sklearn.metrics import f1_score
from autoencoder_tabular import Autoencoder
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from vae_tabular import VAE
from os.path import isfile, join
import os
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)

train_frac = 0.8
folds = 5
num_classes = 2
dataset = 'data_project_name'
max_depth = None

common_columns = ['sex', 'year_of_diagnosis', 'age_of_diagnosis', 'race_-1', 'race_0', 'race_1', 'race_2', 'race_3',
                  'icd_10_0', 'icd_10_1', 'icd_10_2', 'icd_10_3', 'icd_10_4', 'icd_10_5', 'histology_-1', 'histology_0',
                  'histology_1', 'histology_2', 'histology_3', 'histology_4', 'histology_5', 'histology_6',
                  'histology_7', 'histology_8', 'laterality_0', 'laterality_1', 'laterality_2']


def process(a_dataset, b_dataset, epochs=1000):
    a_df = pd.read_csv(a_dataset)
    a_df = a_df.sample(frac=1.0)
    a_df.reset_index(drop=True, inplace=True)
    X_a = a_df.drop(['label'], axis=1)
    y_a = a_df['label']

    b_df = pd.read_csv(b_dataset)
    b_df = b_df.sample(frac=1.0)
    b_df.reset_index(drop=True, inplace=True)
    X_b = b_df.drop(['label'], axis=1)
    y_b = b_df['label']

    a_scaler = MinMaxScaler()
    a_features = a_scaler.fit_transform(X_a)
    X_a = pd.DataFrame(a_features, columns=X_a.columns)

    b_scaler = MinMaxScaler()
    b_features = b_scaler.fit_transform(X_b)
    X_b = pd.DataFrame(b_features, columns=X_b.columns)

    a_train_features = X_a[:int(len(X_a) * train_frac)]
    a_train_features.reset_index(drop=True, inplace=True)
    a_train_labels = y_a[:int(len(y_a) * train_frac)]
    a_train_labels.reset_index(drop=True, inplace=True)

    x_test_a = X_a[int(len(X_a) * train_frac):]
    y_test_a = y_a[int(len(y_a) * train_frac):]
    x_test_ca = x_test_a[common_columns]
    y_test_ca = y_test_a

    b_train_features = X_b[:int(len(X_b) * train_frac)]
    b_train_features.reset_index(drop=True, inplace=True)
    b_train_labels = y_b[:int(len(y_b) * train_frac)]
    b_train_labels.reset_index(drop=True, inplace=True)

    x_test_b = X_b[int(len(X_b) * train_frac):]
    y_test_b = y_b[int(len(y_b) * train_frac):]
    x_test_cb = x_test_b[common_columns]
    y_test_cb = y_test_b

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    a_indices = []
    for _, val_index in skf.split(a_train_features, a_train_labels):
        a_indices.append(val_index)

    b_indices = []
    for _, val_index in skf.split(b_train_features, b_train_labels):
        b_indices.append(val_index)

    results = []
    for fold_index in range(folds):
        current_a_folds = [x for x in range(folds)]
        current_a_folds.remove(fold_index)
        a_train_indices = np.concatenate([a_indices[i] for i in current_a_folds])
        a_val_indices = a_indices[fold_index]

        current_b_folds = [x for x in range(folds)]
        current_b_folds.remove(fold_index)
        b_train_indices = np.concatenate([b_indices[i] for i in current_b_folds])
        b_val_indices = b_indices[fold_index]

        x_train_a = a_train_features.iloc[a_train_indices]
        y_train_a = a_train_labels.iloc[a_train_indices]
        x_val_a = a_train_features.iloc[a_val_indices]
        y_val_a = a_train_labels.iloc[a_val_indices]

        x_train_ca = a_train_features.iloc[a_train_indices][common_columns]
        y_train_ca = y_train_a
        x_val_ca = a_train_features.iloc[a_val_indices][common_columns]
        y_val_ca = y_val_a

        x_train_b = b_train_features.iloc[b_train_indices]
        y_train_b = b_train_labels.iloc[b_train_indices]
        x_val_b = b_train_features.iloc[b_val_indices]
        y_val_b = b_train_labels.iloc[b_val_indices]

        x_train_cb = b_train_features.iloc[b_train_indices][common_columns]
        y_train_cb = y_train_b
        x_val_cb = b_train_features.iloc[b_val_indices][common_columns]
        y_val_cb = y_val_b

        cls_a = RandomForestClassifier(max_depth=max_depth)
        cls_a.fit(x_train_a, y_train_a)
        results_a = cls_a.predict(x_test_a)
        f1_a = f1_score(y_test_a, results_a, average='weighted')
        print(f'F1 A: {f1_a}')
        results.append([dataset, 'base', 'A', fold_index, f1_a])

        ################

        # ~~~ CB-B ~~~
        cb_b_model = Autoencoder(x_train_cb, y_train_cb, x_val_cb, y_val_cb, x_train_b, y_train_b, x_val_b, y_val_b,
                                 data_mode_1='cb', data_mode_2='b',
                                 dataset=dataset, num_classes=num_classes, fold=fold_index)
        cb_b_autoencoder = cb_b_model.get_autoencoder_model()

        # Combine Data #
        decoded_cb_b_train = cb_b_autoencoder.predict(x_train_ca)
        decoded_cb_b_val = cb_b_autoencoder.predict(x_val_ca)
        decoded_cb_b_test = cb_b_autoencoder.predict(x_test_ca)

        x_train_b_update = decoded_cb_b_train[:, len(common_columns):]
        x_val_b_update = decoded_cb_b_val[:, len(common_columns):]
        x_test_b_update = decoded_cb_b_test[:, len(common_columns):]

        combined_data_train_a = np.append(x_train_a, x_train_b_update, axis=1)
        combined_data_val_a = np.append(x_val_a, x_val_b_update, axis=1)
        combined_data_test_a = np.append(x_test_a, x_test_b_update, axis=1)
        #################

        cls_a_b = RandomForestClassifier(max_depth=max_depth)
        cls_a_b.fit(combined_data_train_a, y_train_a)
        results_a_b = cls_a_b.predict(combined_data_test_a)
        f1_a_b = f1_score(y_test_a, results_a_b, average='weighted')
        print(f'F1 A-B: {f1_a_b}')
        results.append([dataset, 'AE', 'A + B*', fold_index, f1_a_b])

        vae_cb_b = VAE(x_train_cb.shape[1], x_train_b.shape[1])
        x_train_b_vae = x_train_b
        x_test_b_vae = x_test_b
        x_train_cb_vae = x_train_cb
        x_test_cb_vae = x_test_cb

        encoder_path = join(dataset, str(fold_index), 'cb_b_vae_encoder.h5')
        decoder_path = join(dataset, str(fold_index), 'cb_b_vae_decoder.h5')
        if isfile(encoder_path) and isfile(decoder_path):
            vae_cb_b.encoder.load_weights(encoder_path)
            vae_cb_b.decoder.load_weights(decoder_path)
        else:
            vae_cb_b.compile(optimizer=keras.optimizers.Adam())
            vae_cb_b.fit(x_test_cb_vae, x_test_b_vae, epochs=epochs, batch_size=4096, shuffle=True,
                    callbacks=[EarlyStopping(monitor='reconstruction_loss', patience=50)])
            os.makedirs(dataset, exist_ok=True)
            vae_cb_b.encoder.save_weights(encoder_path)
            vae_cb_b.decoder.save_weights(decoder_path)

        encoded_cb_b_train = vae_cb_b.encoder(x_train_ca.values)[-1]
        decoded_cb_b_train = vae_cb_b.decoder.predict(encoded_cb_b_train)
        encoded_cb_b_val = vae_cb_b.encoder(x_val_ca.values)[-1]
        decoded_cb_b_val = vae_cb_b.decoder.predict(encoded_cb_b_val)
        encoded_cb_b_test = vae_cb_b.encoder(x_test_ca.values)[-1]
        decoded_cb_b_test = vae_cb_b.decoder.predict(encoded_cb_b_test)

        x_train_b_update = decoded_cb_b_train[:, len(common_columns):]
        x_val_b_update = decoded_cb_b_val[:, len(common_columns):]
        x_test_b_update = decoded_cb_b_test[:, len(common_columns):]

        combined_data_train_a_vae = np.append(x_train_a, x_train_b_update, axis=1)
        combined_data_val_a_vae = np.append(x_val_a, x_val_b_update, axis=1)
        combined_data_test_a_vae = np.append(x_test_a, x_test_b_update, axis=1)

        cls_a_b_vae = RandomForestClassifier(max_depth=max_depth)
        cls_a_b_vae.fit(combined_data_train_a_vae, y_train_a)
        results_a_b_vae = cls_a_b_vae.predict(combined_data_test_a)
        f1_a_b_vae = f1_score(y_test_a, results_a_b_vae, average='weighted')
        print(f'F1 A-B VAE: {f1_a_b_vae}')
        results.append([dataset, 'VAE', 'A + B*', fold_index, f1_a_b_vae])

        ################

        b_model = Autoencoder(x_train_b, y_train_b, x_val_b, y_val_b, num_features=b_features.shape[1], num_classes=num_classes,
                              data_mode_1='b', dataset=dataset, fold=fold_index)

        cls_b = RandomForestClassifier()
        cls_b.fit(x_train_b, y_train_b)
        results_b = cls_b.predict(x_test_b)
        f1_b = f1_score(y_test_b, results_b, average='weighted')
        print(f'F1 B: {f1_b}')
        results.append([dataset, 'base', 'B', fold_index, f1_b])

        ################

        # ~~~ CA-A ~~~
        ca_a_model = Autoencoder(x_train_ca, y_train_ca, x_val_ca, y_val_ca, x_train_a, y_train_a, x_val_a,
                                 y_val_a, data_mode_1='ca', data_mode_2='a', dataset=dataset, num_classes=num_classes,
                                 fold=fold_index)
        ca_a_autoencoder = ca_a_model.get_autoencoder_model()
        ca_a_decoded = ca_a_autoencoder.predict(x_test_ca)

        # Combine Data #
        decoded_ca_a_train = ca_a_autoencoder.predict(x_train_cb)
        decoded_ca_a_val = ca_a_autoencoder.predict(x_val_cb)
        decoded_ca_a_test = ca_a_autoencoder.predict(x_test_cb)

        x_train_a_update = decoded_ca_a_train[:, len(common_columns):]
        x_val_a_update = decoded_ca_a_val[:, len(common_columns):]
        x_test_a_update = decoded_ca_a_test[:, len(common_columns):]

        combined_data_train_b = np.append(x_train_b, x_train_a_update, axis=1)
        combined_data_val_b = np.append(x_val_b, x_val_a_update, axis=1)
        combined_data_test_b = np.append(x_test_b, x_test_a_update, axis=1)
        #################

        cls_b_a = RandomForestClassifier()
        cls_b_a.fit(combined_data_train_b, y_train_b)
        results_b_a = cls_b_a.predict(combined_data_test_b)
        f1_b_a = f1_score(y_test_b, results_b_a, average='weighted')
        print(f'F1 B-A: {f1_b_a}')
        results.append([dataset, 'AE', 'B + A*', fold_index, f1_b_a])

        vae_ca_a = VAE(x_train_ca.shape[1], x_train_a.shape[1])
        x_train_a_vae = x_train_a
        x_test_a_vae = x_test_a
        x_train_ca_vae = x_train_ca
        x_test_ca_vae = x_test_ca

        encoder_path = join(dataset, str(fold_index), 'ca_a_vae_encoder.h5')
        decoder_path = join(dataset, str(fold_index), 'ca_a_vae_decoder.h5')
        if isfile(encoder_path) and isfile(decoder_path):
            vae_ca_a.encoder.load_weights(encoder_path)
            vae_ca_a.decoder.load_weights(decoder_path)
        else:
            vae_ca_a.compile(optimizer=keras.optimizers.Adam())
            vae_ca_a.fit(x_test_ca_vae, x_test_a_vae, epochs=epochs, batch_size=4096, shuffle=True,
                    callbacks=[EarlyStopping(monitor='reconstruction_loss', patience=50)])
            os.makedirs(dataset, exist_ok=True)
            vae_ca_a.encoder.save_weights(encoder_path)
            vae_ca_a.decoder.save_weights(decoder_path)

        encoded_ca_a_train = vae_ca_a.encoder(x_train_cb.values)[-1]
        decoded_ca_a_train = vae_ca_a.decoder.predict(encoded_ca_a_train)
        encoded_ca_a_val = vae_ca_a.encoder(x_val_cb.values)[-1]
        decoded_ca_a_val = vae_ca_a.decoder.predict(encoded_ca_a_val)
        encoded_ca_a_test = vae_ca_a.encoder(x_test_cb.values)[-1]
        decoded_ca_a_test = vae_ca_a.decoder.predict(encoded_ca_a_test)

        x_train_a_update = decoded_ca_a_train[:, len(common_columns):]
        x_val_a_update = decoded_ca_a_val[:, len(common_columns):]
        x_test_a_update = decoded_ca_a_test[:, len(common_columns):]

        combined_data_train_b_vae = np.append(x_train_b, x_train_a_update, axis=1)
        combined_data_val_b_vae = np.append(x_val_b, x_val_a_update, axis=1)
        combined_data_test_b_vae = np.append(x_test_b, x_test_a_update, axis=1)

        cls_b_a_vae = RandomForestClassifier()
        cls_b_a_vae.fit(combined_data_train_b_vae, y_train_b)
        results_b_a_vae = cls_b_a_vae.predict(combined_data_test_b_vae)
        f1_b_a_vae = f1_score(y_test_b, results_b_a_vae, average='weighted')
        print(f'F1 B-A VAE: {f1_b_a_vae}')
        results.append([dataset, 'VAE', 'B + A*', fold_index, f1_b_a_vae])

    df = pd.DataFrame(results)
    df.columns = ['dataset', 'algorithm', 'data_label', 'fold', 'f1_score']
    df.to_csv(join(dataset, 'results.csv'), index=False)


def main():
    process('path_to_datase_a', 'path_to_datase_b')


if __name__ == '__main__':
    main()
