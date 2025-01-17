from data_provider.data_loader import Dataset_Blood, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTH1': Dataset_ETT_hour,
    'ETTH2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'WTH': Dataset_Custom,
    'electricity' : Dataset_Custom,#[321]
    'exchange_rate': Dataset_Custom,#[8]
    'illness': Dataset_Custom,#[7]
    'traffic': Dataset_Custom,#[862]
    'weather': Dataset_Custom,#[21]
    'BLOOD' : Dataset_Blood
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if args.model == 'Arima': 
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = 1  # bsz=1 for evaluation
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size  # bsz for train and valid
            freq = args.freq

    if args.data == 'BLOOD':
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            cut_name = args.cut_name,
            split_name = args.split_name
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    print(flag, len(data_set))
    if len(data_set) < batch_size:
        batch_size = len(data_set)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
