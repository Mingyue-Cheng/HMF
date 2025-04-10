from data_provider.data_loader import Dataset_ETT_hour_all_pre, Dataset_ETT_minute_all_pre, Dataset_Custom_all_pre
from torch.utils.data import DataLoader

data_dict = {
    'ETTH1': Dataset_ETT_hour_all_pre,
    'ETTH2': Dataset_ETT_hour_all_pre,
    'ETTm1': Dataset_ETT_minute_all_pre,
    'ETTm2': Dataset_ETT_minute_all_pre,
    'ECL': Dataset_Custom_all_pre,
    'WTH': Dataset_Custom_all_pre
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

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
