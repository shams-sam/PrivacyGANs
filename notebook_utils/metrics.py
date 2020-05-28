from .eigan import Discriminator
import numpy as np
from .utility import to_numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch


def centralized(
        transform,
        input_size,
        hidden_size,
        output_size,
        X_train,
        X_valid,
        y_train,
        y_valid,
        w,
        device,
        clfs=['mlp', 'logistic', 'svm'],
):

    if not transform:
        def transform(arg):
            return arg

    predicted_tape = {}
    error_rate_tape = {}
    accuracy_tape = {}
    f1_tape = {}

    X_train_ = transform(X_train)
    X_valid_ = transform(X_valid)

    if 'mlp' in clfs:
        mlp = Discriminator(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=2).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([w])).to(device)
        print('='*80)
        print('MLP')
        print('-'*80)
        print("epoch \t mlp_train \t mlp_valid \t v_acc")
        print('-'*80)

        mlp.train()
        n_iter = 1001

        for epoch in range(n_iter):
            y_train_ = mlp(X_train_)
            y_valid_ = mlp(X_valid_)

            mlp_optimizer.zero_grad()
            mlp_train_loss = bce_loss(y_train_, y_train)
            mlp_valid_loss = bce_loss(y_valid_, y_valid)
            mlp_train_loss.backward(retain_graph=True)
            mlp_optimizer.step()

            total = y_valid.size(0)
            predicted = np.argmax(to_numpy(y_valid_), axis=1)
            actuals = np.argmax(to_numpy(y_valid), axis=1)
            correct = (predicted == actuals).sum().item()

            if epoch % 200 != 0:
                continue

            print('{} \t {:.6f} \t {:.6f} \t {:.4f}'.format(
                epoch,
                mlp_train_loss.item(),
                mlp_valid_loss.item(),
                correct/total,
            ))

            predicted = np.argmax(to_numpy(mlp(X_valid_)), axis=1)
            actuals = np.argmax(to_numpy(y_valid), axis=1)
            mlp_cm = confusion_matrix(predicted, actuals)
            error_rate_mlp = (mlp_cm[0, 1] + mlp_cm[1, 0])/mlp_cm.sum()
            correct = (predicted == actuals).sum().item()
            total = y_valid.size(0)

        predicted_tape['mlp'] = predicted
        error_rate_tape['mlp'] = error_rate_mlp
        accuracy_tape['mlp'] = correct/total
        f1_tape['mlp'] = f1_score(predicted, actuals)
        assert error_rate_tape['mlp'] + accuracy_tape['mlp'] == 1
        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['mlp'], error_rate_tape['mlp'], f1_tape['mlp']))

    if 'logistic' in clfs:
        print('='*80)
        print('Logistic Regression')
        print('-'*80)

        # Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(
            to_numpy(X_train_),
            np.argmax(to_numpy(y_train), axis=1), 
            w[np.argmax(to_numpy(y_train), axis=1)]
        )

        y_valid_ = log_reg.predict_proba(to_numpy(X_valid_))
        predicted = np.argmax(y_valid_, axis=1)
        actuals = np.argmax(to_numpy(y_valid), axis=1)
    
        log_reg_cm = confusion_matrix(predicted, actuals)
        error_rate_log_reg = (log_reg_cm[0, 1] + log_reg_cm[1, 0])/log_reg_cm.sum()

        predicted_tape['logistic'] = y_valid_
        error_rate_tape['logistic'] = error_rate_log_reg
        accuracy_tape['logistic'] = accuracy_score(predicted, actuals)
        f1_tape['logistic'] = f1_score(predicted, actuals)
        assert error_rate_tape['logistic'] + accuracy_tape['logistic'] == 1

        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['logistic'],
            error_rate_tape['logistic'],
            f1_tape['logistic'])
        )

    if 'svm' in clfs:
        print('='*80)
        print('SVM')
        print('-'*80)

        # Support Vector Machine
        svm = SVC(probability=True)
        svm.fit(
            to_numpy(X_train_),
            np.argmax(to_numpy(y_train), axis=1),
            w[np.argmax(to_numpy(y_train), axis=1)]
        )

        y_valid_ = svm.predict_proba(to_numpy(X_valid_))
        predicted = np.argmax(y_valid_, axis=1)
        actuals = np.argmax(to_numpy(y_valid), axis=1)

        svm_cm = confusion_matrix(predicted, actuals)
        error_rate_svm = (svm_cm[0, 1] + svm_cm[1, 0])/svm_cm.sum()

        predicted_tape['svm'] = y_valid_
        error_rate_tape['svm'] = error_rate_svm
        accuracy_tape['svm'] = accuracy_score(predicted, actuals)
        f1_tape['svm'] = f1_score(predicted, actuals)
        assert error_rate_tape['svm'] + accuracy_tape['svm'] == 1

        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['svm'], error_rate_tape['svm'], f1_tape['svm'])
        )

    return predicted_tape, error_rate_tape, accuracy_tape, f1_tape


def distributed(
        transforms,
        num_nodes,
        input_size,
        hidden_size,
        output_size,
        X_trains,
        X_valids,
        y_trains,
        y_valids,
        w,
        device,
        clfs=['mlp', 'logistic', 'svm']
):

    if not transforms:
        def transform_(arg):
            return arg
        transforms = [transform_]*num_nodes
    assert len(transforms) == num_nodes

    predicted_tape = {}
    error_rate_tape = {}
    accuracy_tape = {}
    f1_tape = {}

    X_train_ = []
    X_valid_ = []
    for node_idx in range(num_nodes):
        X_train_.append(transforms[node_idx](X_trains[node_idx]))
        X_valid_.append(transforms[node_idx](X_valids[node_idx]))
    X_train_ = torch.cat(X_train_, dim=0)
    X_valid_ = torch.cat(X_valid_, dim=0)

    X_train = torch.cat(X_trains, dim=0)
    X_valid = torch.cat(X_valids, dim=0)

    y_train = torch.cat(y_trains, dim=0)
    y_valid = torch.cat(y_valids, dim=0)

    if 'mlp' in clfs:
        mlp = Discriminator(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=2).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([w])).to(device)
        print('='*80)
        print('MLP')
        print('-'*80)
        print("epoch \t mlp_train \t mlp_valid \t v_acc")
        print('-'*80)

        mlp.train()
        n_iter = 1001

        for epoch in range(n_iter):        
            y_train_ = mlp(X_train_)
            y_valid_ = mlp(X_valid_)

            mlp_optimizer.zero_grad()
            mlp_train_loss = bce_loss(y_train_, y_train)
            mlp_valid_loss = bce_loss(y_valid_, y_valid)
            mlp_train_loss.backward(retain_graph=True)
            mlp_optimizer.step()

            total = y_valid.size(0)
            predicted = np.argmax(to_numpy(y_valid_), axis=1)
            actuals = np.argmax(to_numpy(y_valid), axis=1)
            correct = (predicted == actuals).sum().item()

            if epoch % 200 != 0:
                continue

            print('{} \t {:.6f} \t {:.6f} \t {:.4f}'.format(
                epoch,
                mlp_train_loss.item(),
                mlp_valid_loss.item(),
                correct/total,
            ))

        predicted = np.argmax(to_numpy(mlp(X_valid_)), axis=1)
        actuals = np.argmax(to_numpy(y_valid), axis=1)
        mlp_cm = confusion_matrix(predicted, actuals)
        error_rate_mlp = (mlp_cm[0, 1] + mlp_cm[1, 0])/mlp_cm.sum()
        correct = (predicted == actuals).sum().item()
        total = y_valid.size(0)

        predicted_tape['mlp'] = predicted
        error_rate_tape['mlp'] = error_rate_mlp
        accuracy_tape['mlp'] = correct/total
        f1_tape['mlp'] = f1_score(predicted, actuals)
        assert error_rate_tape['mlp'] + accuracy_tape['mlp'] == 1
        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['mlp'], error_rate_tape['mlp'], f1_tape['mlp']))

    if 'logistic' in clfs:
        print('='*80)
        print('Logistic Regression')
        print('-'*80)

        # Logistic Regression
        X_train
    
        log_reg = LogisticRegression()
        log_reg.fit(
            to_numpy(X_train_), 
            np.argmax(to_numpy(y_train), axis=1), 
            w[np.argmax(to_numpy(y_train), axis=1)]
        )

        y_valid_ = log_reg.predict_proba(to_numpy(X_valid_))
        predicted = np.argmax(y_valid_, axis=1)
        actuals = np.argmax(to_numpy(y_valid), axis=1)

        log_reg_cm = confusion_matrix(predicted, actuals)
        error_rate_log_reg = (log_reg_cm[0, 1] + log_reg_cm[1, 0])/log_reg_cm.sum()
    
        predicted_tape['logistic'] = y_valid_
        error_rate_tape['logistic'] = error_rate_log_reg
        accuracy_tape['logistic'] = accuracy_score(predicted, actuals)
        f1_tape['logistic'] = f1_score(predicted, actuals)
        assert error_rate_tape['logistic'] + accuracy_tape['logistic'] == 1

        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['logistic'], error_rate_tape['logistic'], f1_tape['logistic']))

    if 'svm' in clfs:
        print('='*80)
        print('SVM')
        print('-'*80)
    
        # Support Vector Machine
        svm = SVC(probability=True)
        svm.fit(
            to_numpy(X_train_),
            np.argmax(to_numpy(y_train), axis=1), 
            w[np.argmax(to_numpy(y_train), axis=1)])

        y_valid_ = svm.predict_proba(to_numpy(X_valid_))
        predicted = np.argmax(y_valid_, axis=1)
        actuals = np.argmax(to_numpy(y_valid), axis=1)

        svm_cm = confusion_matrix(predicted, actuals)
        error_rate_svm = (svm_cm[0, 1] + svm_cm[1, 0])/svm_cm.sum()
    
        predicted_tape['svm'] = y_valid_
        error_rate_tape['svm'] = error_rate_svm
        accuracy_tape['svm'] = accuracy_score(predicted, actuals)
        f1_tape['svm'] = f1_score(predicted, actuals)
        assert error_rate_tape['svm'] + accuracy_tape['svm'] == 1

        print('Accuracy: {}\nError Rate: {}\nF1 Score: {}'.format(
            accuracy_tape['svm'], error_rate_tape['svm'], f1_tape['svm']))

    return predicted_tape, error_rate_tape, accuracy_tape, f1_tape
