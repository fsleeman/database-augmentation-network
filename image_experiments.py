from os.path import join

import numpy as np
from keras.datasets import mnist, cifar10
from pandas import DataFrame
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import utils
from autoencoder import Autoencoder
from classifier import Classifier
from vae import VAE


def process(dataset, common_cols, epochs=250, folds=5):
    if dataset == 'mnist':
        (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
        x_dim, y_dim, channels = 28, 28, 1
        num_classes = 10
    elif dataset == 'cifar10':
        (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
        x_dim, y_dim, channels = 28, 28, 1
        num_classes = 10
    else:
        assert f'Dataset {dataset} is not valid'
        return

    x_dim_keep = int(x_dim / 2)
    x_test = x_test.astype('float32') / 255.

    test_results = []

    skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)
    fold_index = 0
    for train_index, val_index in skf.split(x_train_total, y_train_total):
        fold_index += 1

        x_train = x_train_total[train_index]
        y_train = y_train_total[train_index]

        x_train = x_train.astype('float32') / 255.
        ################
        # ~~~ CA ~~~
        (x_train_ca, y_train_ca), (x_val_ca, y_val_ca) = utils.get_data(x_train, y_train, side='ca',
                                                                        x_dim_keep=x_dim_keep,
                                                                        common_cols=common_cols, dataset=dataset)
        (x_test_ca, y_test_ca) = utils.get_data(x_test, y_test, side='ca', is_test=True, x_dim_keep=x_dim_keep,
                                                common_cols=common_cols,
                                                dataset=dataset, train_frac=1.0)
        # ~~~ CB ~~~
        (x_train_cb, y_train_cb), (x_val_cb, y_val_cb) = utils.get_data(x_train, y_train, side='cb',
                                                                        x_dim_keep=x_dim_keep,
                                                                        common_cols=common_cols, dataset=dataset)
        (x_test_cb, y_test_cb) = utils.get_data(x_test, y_test, side='cb', is_test=True, x_dim_keep=x_dim_keep,
                                                common_cols=common_cols,
                                                dataset=dataset, train_frac=1.0)
        # ~~~ A ~~~
        (x_train_a, y_train_a), (x_val_a, y_val_a) = utils.get_data(x_train, y_train, side='a', x_dim_keep=x_dim_keep,
                                                                    common_cols=common_cols, dataset=dataset)

        (x_test_a, y_test_a) = utils.get_data(x_test, y_test, side='a', is_test=True, x_dim_keep=x_dim_keep,
                                              common_cols=common_cols,
                                              dataset=dataset, train_frac=1.0)
        # ~~~ B ~~~
        (x_train_b, y_train_b), (x_val_b, y_val_b) = utils.get_data(x_train, y_train, side='b', x_dim_keep=x_dim_keep,
                                                                    common_cols=common_cols, dataset=dataset)
        (x_test_b, y_test_b) = utils.get_data(x_test, y_test, side='b', is_test=True, x_dim_keep=x_dim_keep,
                                              common_cols=common_cols,
                                              dataset=dataset, train_frac=1.0)

        x_train_cb_b_ae2 = x_train_cb.reshape((len(x_train_cb), x_dim, y_dim, channels))
        x_train_b_ae2 = x_train_b.reshape((len(x_train_b), x_dim, y_dim, channels))

        '''~~~~~~~~~~~~~~~~~ A ~~~~~~~~~~~~~~~~~'''
        a_cls = Classifier(x_train_a, y_train_a, X_val=x_val_a, y_val=y_val_a, input_shape=(x_dim, y_dim, channels),
                           num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                           name='a', fold=fold_index, epochs=epochs)

        a_test_results = np.argmax(a_cls.model.predict(x_test_a), axis=1)
        a_f1 = f1_score(y_test_a, a_test_results, average='weighted')
        test_results.append([dataset, common_cols, 'base', 'A', fold_index, a_f1])
        print(f'Base f1 score A: {a_f1}')

        '''~~~~~~~~~~~~~~~~~ B ~~~~~~~~~~~~~~~~~'''
        b_cls = Classifier(x_train_b, y_train_b, X_val=x_val_b, y_val=y_val_b, input_shape=(x_dim, y_dim, channels),
                           num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                           name='b', fold=fold_index, epochs=epochs)

        b_test_results = np.argmax(b_cls.model.predict(x_test_b), axis=1)
        b_f1 = f1_score(y_test_b, b_test_results, average='weighted')
        test_results.append([dataset, common_cols, 'base', 'B', fold_index, b_f1])
        print(f'Base f1 score B: {b_f1}')

        # ~~~~~ Networks ~~~~~
        # ~~~ CB-B ~~~
        ae2_cb_b = Autoencoder(x_train_1=x_train_cb_b_ae2, x_train_2=x_train_b_ae2, data_mode_1='cb', data_mode_2='b',
                       dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                       epochs=epochs)

        x_train_ca_a_ae2 = x_train_ca.reshape((len(x_train_ca), x_dim, y_dim, channels))
        x_train_a_ae2 = x_train_a.reshape((len(x_train_a), x_dim, y_dim, channels))

        # ~~~ CA-A ~~~
        ae2_ca_a = Autoencoder(x_train_1=x_train_ca_a_ae2, x_train_2=x_train_a_ae2, data_mode_1='ca', data_mode_2='a',
                       dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                       epochs=epochs)

        # ~~~~ CB-B AE2 ~~~~
        encoded_imgs_cb_b_ae2_train = ae2_cb_b.encoder(x_train_ca)
        decoded_imgs_cb_b_ae2_train = ae2_cb_b.decoder.predict(encoded_imgs_cb_b_ae2_train)
        x_train_b_ae2_update = decoded_imgs_cb_b_ae2_train.copy()
        x_train_b_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_b_ae2_update = np.squeeze(x_train_b_ae2_update)
        combined_data_train_a_ae2 = x_train_a + x_train_b_ae2_update

        encoded_imgs_cb_b_ae2_val = ae2_cb_b.encoder(x_val_ca)
        decoded_imgs_cb_b_ae2_val = ae2_cb_b.decoder.predict(encoded_imgs_cb_b_ae2_val)
        x_val_b_ae2_update = decoded_imgs_cb_b_ae2_val.copy()
        x_val_b_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_b_ae2_update = np.squeeze(x_val_b_ae2_update)
        combined_data_val_a_ae2 = x_val_a + x_val_b_ae2_update

        cb_b_ae2_cls = Classifier(combined_data_train_a_ae2, y_train_a,
                                  X_val=combined_data_val_a_ae2, y_val=y_val_a,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='cb_b_ae2', fold=fold_index, epochs=epochs)

        encoded_imgs_cb_b_ae2_test = ae2_cb_b.encoder(x_test_ca)
        decoded_imgs_cb_b_ae2_test = ae2_cb_b.decoder.predict(encoded_imgs_cb_b_ae2_test)
        x_test_b_ae2_update = decoded_imgs_cb_b_ae2_test.copy()
        x_test_b_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_b_ae2_update = np.squeeze(x_test_b_ae2_update)
        combined_data_test_a_ae2 = x_test_a + x_test_b_ae2_update

        ae2_blend_results_a = np.argmax(cb_b_ae2_cls.model.predict(combined_data_test_a_ae2), axis=1)
        ab_ae2_blend_f1 = f1_score(y_test_a, ae2_blend_results_a, average='weighted')
        test_results.append([dataset, common_cols, 'AE2', 'A + B*', fold_index, ab_ae2_blend_f1])
        print(f'AE2 (A + B*) Blend data F1: {ab_ae2_blend_f1}')

        # ~~~~ CA-A AE2 ~~~~
        encoded_imgs_ca_a_ae2_train = ae2_ca_a.encoder(x_train_cb)
        decoded_imgs_ca_a_ae2_train = ae2_ca_a.decoder.predict(encoded_imgs_ca_a_ae2_train)
        x_train_a_ae2_update = decoded_imgs_ca_a_ae2_train.copy()
        x_train_a_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_a_ae2_update = np.squeeze(x_train_a_ae2_update)
        combined_data_train_b_ae2 = x_train_b + x_train_a_ae2_update

        encoded_imgs_ca_a_ae2_val = ae2_ca_a.encoder(x_val_cb)
        decoded_imgs_ca_a_ae2_val = ae2_ca_a.decoder.predict(encoded_imgs_ca_a_ae2_val)
        x_val_a_ae2_update = decoded_imgs_ca_a_ae2_val.copy()
        x_val_a_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_a_ae2_update = np.squeeze(x_val_a_ae2_update)
        combined_data_val_b_ae2 = x_val_b + x_val_a_ae2_update

        ca_a_ae2_cls = Classifier(combined_data_train_b_ae2, y_train_b,
                                  X_val=combined_data_val_b_ae2, y_val=y_val_b,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='ca_a_ae2', fold=fold_index, epochs=epochs)

        encoded_imgs_ca_a_ae2_test = ae2_ca_a.encoder(x_test_cb)
        decoded_imgs_ca_a_ae2_test = ae2_ca_a.decoder.predict(encoded_imgs_ca_a_ae2_test)
        x_test_a_ae2_update = decoded_imgs_ca_a_ae2_test.copy()
        x_test_a_ae2_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_a_ae2_update = np.squeeze(x_test_a_ae2_update)
        combined_data_test_b_ae2 = x_test_b + x_test_a_ae2_update

        ae2_blend_results_b = np.argmax(ca_a_ae2_cls.model.predict(combined_data_test_b_ae2), axis=1)
        ba_ae2_blend_f1 = f1_score(y_test_b, ae2_blend_results_b, average='weighted')
        test_results.append([dataset, common_cols, 'AE2', 'B + A*', fold_index, ba_ae2_blend_f1])
        print(f'AE2 (B + A*) Blend data F1: {ba_ae2_blend_f1}')

        # ~~~~~~~~~ VAE ~~~~~~~~~~~~~~~~~
        x_train_cb_b_vae = x_train_cb.reshape((len(x_train_cb), x_dim, y_dim, channels))
        x_train_b_vae = x_train_b.reshape((len(x_train_b), x_dim, y_dim, channels))

        vae2_cb_b = VAE(x_train_1=x_train_cb_b_vae, x_train_2=x_train_b_vae, data_mode_1='cb', data_mode_2='b',
                        dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                        epochs=epochs)

        x_train_ca_a_vae = x_train_ca.reshape((len(x_train_ca), x_dim, y_dim, channels))
        x_train_a_vae = x_train_a.reshape((len(x_train_a), x_dim, y_dim, channels))

        vae2_ca_a = VAE(x_train_1=x_train_ca_a_vae, x_train_2=x_train_a_vae, data_mode_1='ca', data_mode_2='a',
                        dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                        epochs=epochs)

        ''' CB-B - VAE '''
        encoded_imgs_cb_b_vae2_train = vae2_cb_b.encoder(x_train_ca)[-1]
        decoded_imgs_cb_b_vae2_train = vae2_cb_b.decoder.predict(encoded_imgs_cb_b_vae2_train)
        x_train_b_vae_update = decoded_imgs_cb_b_vae2_train.copy()
        x_train_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_b_vae_update = np.squeeze(x_train_b_vae_update)
        combined_data_train_a_vae = x_train_a + x_train_b_vae_update

        encoded_imgs_cb_b_vae2_val = vae2_cb_b.encoder(x_val_ca)[-1]
        decoded_imgs_cb_b_vae2_val = vae2_cb_b.decoder.predict(encoded_imgs_cb_b_vae2_val)
        x_val_b_vae_update = decoded_imgs_cb_b_vae2_val.copy()
        x_val_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_b_vae_update = np.squeeze(x_val_b_vae_update)
        combined_data_val_a_vae = x_val_a + x_val_b_vae_update

        cb_b_vae_cls = Classifier(combined_data_train_a_vae, y_train_a,
                                  X_val=combined_data_val_a_vae, y_val=y_val_a,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='cb_b_vae2', fold=fold_index, epochs=epochs)

        encoded_imgs_cb_b_vae2_test = vae2_cb_b.encoder(x_test_ca)[-1]
        decoded_imgs_cb_b_vae2_test = vae2_cb_b.decoder.predict(encoded_imgs_cb_b_vae2_test)
        x_test_b_vae_update = decoded_imgs_cb_b_vae2_test.copy()
        x_test_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_b_vae_update = np.squeeze(x_test_b_vae_update)
        combined_data_test_a_vae = x_test_a + x_test_b_vae_update

        vae_blend_results_a = np.argmax(cb_b_vae_cls.model.predict(combined_data_test_a_vae), axis=1)
        ab_vae_blend_f1 = f1_score(y_test_a, vae_blend_results_a, average='weighted')
        test_results.append([dataset, common_cols, 'VAE', 'A + B*', fold_index, ab_vae_blend_f1])
        print(f'VAE (A + B*) Blend data F1: {ab_vae_blend_f1}')

        ''' CA-A - VAE '''
        encoded_imgs_ca_a_vae_train = vae2_ca_a.encoder(x_train_cb)[-1]
        decoded_imgs_ca_a_vae_train = vae2_ca_a.decoder.predict(encoded_imgs_ca_a_vae_train)
        x_train_a_vae_update = decoded_imgs_ca_a_vae_train.copy()
        x_train_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_a_vae_update = np.squeeze(x_train_a_vae_update)
        combined_data_train_b_vae = x_train_b + x_train_a_vae_update

        encoded_imgs_cb_b_vae_val = vae2_cb_b.encoder(x_val_cb)[-1]
        decoded_imgs_ca_a_vae_val = vae2_ca_a.decoder.predict(encoded_imgs_cb_b_vae_val)
        x_val_a_vae_update = decoded_imgs_ca_a_vae_val.copy()
        x_val_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_a_vae_update = np.squeeze(x_val_a_vae_update)
        combined_data_val_b_vae = x_val_b + x_val_a_vae_update

        ca_a_vae_cls = Classifier(combined_data_train_b_vae, y_train_b,
                                  X_val=combined_data_val_b_vae, y_val=y_val_b,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='ca_a_vae2', fold=fold_index, epochs=epochs)

        encoded_imgs_ca_a_vae_test = vae2_ca_a.encoder(x_test_cb)[-1]
        decoded_imgs_ca_a_vae_test = vae2_ca_a.decoder.predict(encoded_imgs_ca_a_vae_test)
        x_test_a_vae_update = decoded_imgs_ca_a_vae_test.copy()
        x_test_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_a_vae_update = np.squeeze(x_test_a_vae_update)
        combined_data_test_b_vae = x_test_b + x_test_a_vae_update

        vae_blend_results_b = np.argmax(ca_a_vae_cls.model.predict(combined_data_test_b_vae), axis=1)
        ba_vae_blend_f1 = f1_score(y_test_b, vae_blend_results_b, average='weighted')
        test_results.append([dataset, common_cols, 'VAE', 'B + A*', fold_index, ba_vae_blend_f1])
        print(f'VAE (B + A*) Blend data F1: {ba_vae_blend_f1}')

        utils.plot_multiple_sources(
            [x_test, x_test_a, combined_data_test_a_ae2, combined_data_test_a_vae, x_test_b, combined_data_test_b_ae2,
             combined_data_test_b_vae],
            [y_train_a, y_train_a, y_train_a, y_train_a], 10)

        '''~~~~~~~~~~~~~~~~~ CB-B ~~~~~~~~~~~~~~~~~'''
        ''' CA-A - AE '''
        ca_a_autoencoder_model = Autoencoder(x_train_ca, x_val_ca, x_train_a, x_val_a, data_mode_1='ca',
                                     data_mode_2='a',
                                     dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim),
                                     fold=fold_index, epochs=epochs)
        ca_a_autoencoder = ca_a_autoencoder_model.get_autoencoder_model()

        ''' CB-B - AE '''
        cb_b_autoencoder_model = Autoencoder(x_train_cb, x_val_cb, x_train_b, x_val_b, data_mode_1='cb',
                                     data_mode_2='b',
                                     dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim),
                                     fold=fold_index, epochs=epochs)
        cb_b_autoencoder = cb_b_autoencoder_model.get_autoencoder_model()

        # Combine Data #
        decoded_imgs_cb_b_ae_train = cb_b_autoencoder.predict(x_train_ca)
        decoded_imgs_cb_b_ae_val = cb_b_autoencoder.predict(x_val_ca)
        decoded_imgs_cb_b_ae_test = cb_b_autoencoder.predict(x_test_ca)

        x_train_b_ae_update = decoded_imgs_cb_b_ae_train.copy()
        x_train_b_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_b_ae_update = np.squeeze(x_train_b_ae_update)

        x_val_b_ae_update = decoded_imgs_cb_b_ae_val.copy()
        x_val_b_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_b_ae_update = np.squeeze(x_val_b_ae_update)

        x_test_b_ae_update = decoded_imgs_cb_b_ae_test.copy()
        x_test_b_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_b_ae_update = np.squeeze(x_test_b_ae_update)

        combined_data_train_a_ae = x_train_a + x_train_b_ae_update
        combined_data_val_a_ae = x_val_a + x_val_b_ae_update
        combined_data_test_a_ae = x_test_a + x_test_b_ae_update
        #################
        cb_b_ae_cls = Classifier(combined_data_train_a_ae, y_train_a, X_val=combined_data_val_a_ae, y_val=y_val_a,
                                 input_shape=(x_dim, y_dim, channels),
                                 num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                 name='cb_b_ae', fold=fold_index, epochs=epochs)

        combined_data_test_a_ae = combined_data_test_a_ae.reshape((len(x_test_a), x_dim, y_dim, channels))
        ae_results_cb_b = np.argmax(cb_b_ae_cls.model.predict(combined_data_test_a_ae), axis=1)

        f1_ae_cb_b = f1_score(y_test_a, ae_results_cb_b, average='weighted')
        test_results.append([dataset, common_cols, 'AE', 'A + B*', fold_index, f1_ae_cb_b])
        print(f'AE (A + B*) Blend data F1: {f1_ae_cb_b}')

        utils.plot_multiple_sources(
            [x_test, x_test_a, combined_data_test_a_ae],
            [y_train_a, y_train_a, y_train_a], 10)

        ''' CB-B - VAE '''
        x_train_cb_b_vae = x_train_cb.reshape((len(x_train_cb), x_dim, y_dim, channels))
        x_train_b_vae = x_train_b.reshape((len(x_train_b), x_dim, y_dim, channels))

        vae_cb_b = VAE(x_train_1=x_train_cb_b_vae, x_train_2=x_train_b_vae, data_mode_1='cb', data_mode_2='b',
                       dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                       epochs=epochs)

        encoded_imgs_cb_b_vae_train = vae_cb_b.encoder(x_train_ca)[-1]
        decoded_imgs_cb_b_vae_train = vae_cb_b.decoder.predict(encoded_imgs_cb_b_vae_train)
        x_train_b_vae_update = decoded_imgs_cb_b_vae_train.copy()
        x_train_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_b_vae_update = np.squeeze(x_train_b_vae_update)
        combined_data_train_a_vae = x_train_a + x_train_b_vae_update

        encoded_imgs_cb_b_vae_val = vae_cb_b.encoder(x_val_ca)[-1]
        decoded_imgs_cb_b_vae_val = vae_cb_b.decoder.predict(encoded_imgs_cb_b_vae_val)
        x_val_b_vae_update = decoded_imgs_cb_b_vae_val.copy()
        x_val_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_b_vae_update = np.squeeze(x_val_b_vae_update)
        combined_data_val_a_vae = x_val_a + x_val_b_vae_update

        cb_b_vae_cls = Classifier(combined_data_train_a_vae, y_train_a,
                                  X_val=combined_data_val_a_vae, y_val=y_val_a,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='cb_b_vae', fold=fold_index, epochs=epochs)

        encoded_imgs_cb_b_vae_test = vae_cb_b.encoder(x_test_ca)[-1]
        decoded_imgs_cb_b_vae_test = vae_cb_b.decoder.predict(encoded_imgs_cb_b_vae_test)
        x_test_b_vae_update = decoded_imgs_cb_b_vae_test.copy()
        x_test_b_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_b_vae_update = np.squeeze(x_test_b_vae_update)
        combined_data_test_a_vae = x_test_a + x_test_b_vae_update

        vae_blend_results_a = np.argmax(cb_b_vae_cls.model.predict(combined_data_test_a_vae), axis=1)
        ab_vae_blend_f1 = f1_score(y_test_a, vae_blend_results_a, average='weighted')
        test_results.append([dataset, common_cols, 'VAE', 'A + B*', fold_index, ab_vae_blend_f1])
        print(f'VAE (A + B*) Blend data F1: {ab_vae_blend_f1}')

        '''~~~~~~~~~~~~~~~~~ CA-A ~~~~~~~~~~~~~~~~~'''
        # Combine Data #
        decoded_imgs_ca_a_ae_train = ca_a_autoencoder.predict(x_train_cb)
        decoded_imgs_ca_a_ae_val = ca_a_autoencoder.predict(x_val_cb)
        decoded_imgs_ca_a_ae_test = ca_a_autoencoder.predict(x_test_cb)

        x_train_a_ae_update = decoded_imgs_ca_a_ae_train.copy()
        x_train_a_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_a_ae_update = np.squeeze(x_train_a_ae_update)

        x_val_a_ae_update = decoded_imgs_ca_a_ae_val.copy()
        x_val_a_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_a_ae_update = np.squeeze(x_val_a_ae_update)

        x_test_a_ae_update = decoded_imgs_ca_a_ae_test.copy()
        x_test_a_ae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_a_ae_update = np.squeeze(x_test_a_ae_update)

        combined_data_train_b_ae = x_train_b + x_train_a_ae_update
        combined_data_val_b_ae = x_val_b + x_val_a_ae_update
        combined_data_test_b_ae = x_test_b + x_test_a_ae_update
        #################
        ca_a_ae_cls = Classifier(combined_data_train_b_ae, y_train_b, X_val=combined_data_val_b_ae, y_val=y_val_b,
                                 input_shape=(x_dim, y_dim, channels),
                                 num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                 name='ca_a_ae', fold=fold_index, epochs=epochs)

        combined_data_test_b_ae = combined_data_test_b_ae.reshape((len(x_test_b), x_dim, y_dim, channels))
        ae_results_ca_a = np.argmax(ca_a_ae_cls.model.predict(combined_data_test_b_ae), axis=1)

        f1_ae_ca_a = f1_score(y_test_b, ae_results_ca_a, average='weighted')
        test_results.append([dataset, common_cols, 'AE', 'B + A*', fold_index, f1_ae_ca_a])
        print(f'AE (B + A*) Blend data F1: {f1_ae_ca_a}')

        ''' CA-A - VAE '''
        x_train_ca_a_vae = x_train_ca.reshape((len(x_train_ca), x_dim, y_dim, channels))
        x_train_a_vae = x_train_a.reshape((len(x_train_a), x_dim, y_dim, channels))

        vae_ca_a = VAE(x_train_1=x_train_ca_a_vae, x_train_2=x_train_a_vae, data_mode_1='ca', data_mode_2='a',
                       dataset=dataset, common_cols=common_cols, shape=(x_dim, y_dim, channels), fold=fold_index,
                       epochs=epochs)

        encoded_imgs_ca_a_vae_train = vae_ca_a.encoder(x_train_cb)[-1]
        decoded_imgs_ca_a_vae_train = vae_ca_a.decoder.predict(encoded_imgs_ca_a_vae_train)
        x_train_a_vae_update = decoded_imgs_ca_a_vae_train.copy()
        x_train_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_train_a_vae_update = np.squeeze(x_train_a_vae_update)
        combined_data_train_b_vae = x_train_b + x_train_a_vae_update

        encoded_imgs_ca_a_vae_val = vae_ca_a.encoder(x_val_cb)[-1]
        decoded_imgs_ca_a_vae_val = vae_ca_a.decoder.predict(encoded_imgs_ca_a_vae_val)
        x_val_a_vae_update = decoded_imgs_ca_a_vae_val.copy()
        x_val_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_val_a_vae_update = np.squeeze(x_val_a_vae_update)
        combined_data_val_b_vae = x_val_b + x_val_a_vae_update

        ca_a_vae_cls = Classifier(combined_data_train_b_vae, y_train_b,
                                  X_val=combined_data_val_b_vae, y_val=y_val_b,
                                  input_shape=(x_dim, y_dim, channels),
                                  num_classes=num_classes, dataset=dataset, common_cols=common_cols,
                                  name='ca_a_vae', fold=fold_index, epochs=epochs)

        encoded_imgs_ca_a_vae_test = vae_ca_a.encoder(x_test_cb)[-1]
        decoded_imgs_ca_a_vae_test = vae_ca_a.decoder.predict(encoded_imgs_ca_a_vae_test)
        x_test_a_vae_update = decoded_imgs_ca_a_vae_test.copy()
        x_test_a_vae_update[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols] = 0
        x_test_a_vae_update = np.squeeze(x_test_a_vae_update)
        combined_data_test_b_vae = x_test_b + x_test_a_vae_update

        vae_blend_results_b = np.argmax(ca_a_vae_cls.model.predict(combined_data_test_b_vae), axis=1)
        ba_vae_blend_f1 = f1_score(y_test_b, vae_blend_results_b, average='weighted')
        test_results.append([dataset, common_cols, 'VAE', 'B + A*', fold_index, ba_vae_blend_f1])
        print(f'VAE (B + A*) Blend data F1: {ba_vae_blend_f1}')

        utils.plot_multiple_sources(
            [x_test, x_test_a, combined_data_test_a_ae, combined_data_test_a_vae, x_test_b, combined_data_test_b_ae,
             combined_data_test_b_vae],
            [y_train_a, y_train_a, y_train_a, y_train_a], 10)

    df = DataFrame(test_results)
    df.to_csv(join(dataset + '_' + str(common_cols), 'results.csv'),
              header=['dataset', 'common_cols', 'algorithm',
                      'data_label', 'fold', 'f1_score'], index=False)


def main():
    process('mnist', common_cols=4)


if __name__ == '__main__':
    main()
