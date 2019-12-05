from utils import *

if __name__ == '__main__':
    num_classes = 1
    seed_everything(1234)
    lr = 3e-5
    IMG_SIZE = 256

    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    duplicated_images = pd.read_csv('../data/duplicated_info.csv')
    duplicated_images['id_code'] = duplicated_images['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    train = train.loc[[x not in duplicated_images['id_code'].values for x in train['id_code'].values], :]
    X = train['id_code'].values

    X_train, X_val, y_train, y_val = train_test_split(X, train['diagnosis'].values, test_size=0.002, random_state=42,
                                                      stratify=train['diagnosis'].values)


    print(f"Num train: {len(X_train)}; Num val: {len(X_val)}")
    model = EfficientNet.from_pretrained('efficientnet-b7')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join('best_256.pth'))['model_state_dict'])

    train_dataset = DiabeticDataset(dataset_path='../data/train_images',
                                    labels=y_train,
                                    ids=X_train,
                                    albumentations_tr=aug_train(IMG_SIZE),
                                    extens='png')
    val_dataset = DiabeticDataset(dataset_path='../data/train_images',
                                    labels=y_val,
                                    ids=X_val,
                                    albumentations_tr=aug_val(IMG_SIZE),
                                    extens='png')

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    weight = dict(zip(np.unique(y_train), weight))
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(train_dataset,
                                num_workers=16,
                                pin_memory=False,
                                batch_size=16,
                                shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=False,
                            batch_size=16)
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader
    runner = SupervisedRunner()
    logdir = f"logs/efficient_net_b7_regression_train"

    print('Training only head for 5 epochs')
    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    num_epochs = 5
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            QuadraticKappScoreMetricCallback(),
            MSECallback(),
            MAECallback(),
            MixupCallback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )

    print('Train whole net for 10 epochs')
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 10
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            QuadraticKappScoreMetricCallback(),
            MSECallback(),
            MAECallback(),
            MixupCallback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )

    print('Training only head for 5 epochs without mixup')
    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 5
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            QuadraticKappScoreMetricCallback(),
            MSECallback(),
            MAECallback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )
    
    print('Train whole net for 20 epochs without mixup')
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 20
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            QuadraticKappScoreMetricCallback(),
            MSECallback(),
            MAECallback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )