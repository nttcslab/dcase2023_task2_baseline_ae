import torch
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
import sys

from datasets.loader_common import get_machine_type_dict
from datasets.dcase_dcase202x_t2_loader import DCASE202XT2Loader

class DCASE202XT2(object):
    def __init__(self, args):
        self.width   = args.frames
        self.height  = args.n_mels
        self.channel = 1
        self.input_dim = self.width*self.height*self.channel
        shuffle = args.shuffle
        batch_sampler = None
        batch_size = args.batch_size
        print("input dim: %d" % (self.input_dim))

        dataset_name = args.dataset[:11]
        machine_type = args.dataset[11:]
        if args.eval:
            data_path = f'{args.dataset_directory}/{dataset_name.lower()}/eval_data/'
            data_type = "eval"
        elif args.dev:
            data_path = f'{args.dataset_directory}/{dataset_name.lower()}/dev_data/'
            data_type = "dev"
        else:
            print("incorrect argument")
            print("please set option argument '--dev' or '--eval'")
            sys.exit()

        self.machine_type_dict = get_machine_type_dict(dataset_name, mode=args.dev)["machine_type"]
        self.section_id_list = self.machine_type_dict[machine_type][data_type]
        self.num_classes = len(self.section_id_list)
        print("num classes: %d" % (self.num_classes))
        self.id_list = [int(machine_id) for machine_id in self.section_id_list]
        section_keyword = get_machine_type_dict(dataset_name, mode=args.dev)["section_keyword"]
        train_data = DCASE202XT2Loader(
                data_path,
                dataset_name=dataset_name,
                section_keyword=section_keyword,
                machine_type=machine_type,
                train=True,
                section_ids=self.section_id_list,
                frames=args.frames,
                n_mels=args.n_mels,
                frame_hop_length=args.frame_hop_length,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                power=args.power,
                fmax=args.fmax,
                fmin=args.fmin,
                win_length=args.win_length,
                data_type=data_type,
                use_id=args.use_ids,
                is_auto_download=args.is_auto_download,
                )

        train_index, valid_index = train_test_split(range(len(train_data)), test_size=args.validation_split)
        self.train_dataset = Subset(train_data, train_index)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler,
        )
        self.valid_dataset   = Subset(train_data, valid_index)
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size, shuffle=False, batch_sampler=batch_sampler,
        )

        self.test_loader = []
        if args.train_only:
            return
        for id in self.section_id_list:
           _test_loader = DCASE202XT2Loader(
                data_path,
                dataset_name=dataset_name,
                section_keyword=section_keyword,
                machine_type=machine_type,
                train=False,
                section_ids=[id],
                frames=args.frames,
                n_mels=args.n_mels,
                frame_hop_length=args.frame_hop_length,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                power=args.power,
                fmax=args.fmax,
                fmin=args.fmin,
                win_length=args.win_length,
                data_type=data_type,
                is_auto_download=args.is_auto_download,
           )

           self.test_loader.append(
                torch.utils.data.DataLoader(
                    _test_loader,
                    batch_size=_test_loader.n_vectors_ea_file, shuffle=False
                )
           )
           self.mode = args.dev or _test_loader.mode


class Datasets:
    DatasetsDic = {
        'DCASE2024T23DPrinter':DCASE202XT2,
        'DCASE2024T2AirCompressor':DCASE202XT2,
        'DCASE2024T2Scanner':DCASE202XT2,
        'DCASE2024T2ToyCircuit':DCASE202XT2,
        'DCASE2024T2HoveringDrone':DCASE202XT2,
        'DCASE2024T2HairDryer':DCASE202XT2,
        'DCASE2024T2ToothBrush':DCASE202XT2,
        'DCASE2024T2RoboticArm':DCASE202XT2,
        'DCASE2024T2BrushlessMotor':DCASE202XT2,
        'DCASE2024T2ToyCar':DCASE202XT2,
        'DCASE2024T2ToyTrain':DCASE202XT2,
        'DCASE2024T2bearing':DCASE202XT2,
        'DCASE2024T2fan':DCASE202XT2,
        'DCASE2024T2gearbox':DCASE202XT2,
        'DCASE2024T2slider':DCASE202XT2,
        'DCASE2024T2valve':DCASE202XT2,
        'DCASE2023T2bandsaw':DCASE202XT2,
        'DCASE2023T2bearing':DCASE202XT2,
        'DCASE2023T2fan':DCASE202XT2,
        'DCASE2023T2grinder':DCASE202XT2,
        'DCASE2023T2gearbox':DCASE202XT2,
        'DCASE2023T2shaker':DCASE202XT2,
        'DCASE2023T2slider':DCASE202XT2,
        'DCASE2023T2ToyCar':DCASE202XT2,
        'DCASE2023T2ToyDrone':DCASE202XT2,
        'DCASE2023T2ToyNscale':DCASE202XT2,
        'DCASE2023T2ToyTank':DCASE202XT2,
        'DCASE2023T2ToyTrain':DCASE202XT2,
        'DCASE2023T2Vacuum':DCASE202XT2,
        'DCASE2023T2valve':DCASE202XT2,
        'DCASE2022T2bearing':DCASE202XT2,
        'DCASE2022T2fan':DCASE202XT2,
        'DCASE2022T2gearbox':DCASE202XT2,
        'DCASE2022T2slider':DCASE202XT2,
        'DCASE2022T2ToyCar':DCASE202XT2,
        'DCASE2022T2ToyTrain':DCASE202XT2,
        'DCASE2022T2valve':DCASE202XT2,
        'DCASE2021T2fan':DCASE202XT2,
        'DCASE2021T2gearbox':DCASE202XT2,
        'DCASE2021T2pump':DCASE202XT2,
        'DCASE2021T2slider':DCASE202XT2,
        'DCASE2021T2ToyCar':DCASE202XT2,
        'DCASE2021T2ToyTrain':DCASE202XT2,
        'DCASE2021T2valve':DCASE202XT2,
        'DCASE2020T2ToyCar':DCASE202XT2,
        'DCASE2020T2ToyConveyor': DCASE202XT2,
        'DCASE2020T2fan':DCASE202XT2,
        'DCASE2020T2valve':DCASE202XT2,
        'DCASE2020T2slider':DCASE202XT2,
        'DCASE2020T2pump':DCASE202XT2,
    }

    def __init__(self,datasets_str):
        self.data = Datasets.DatasetsDic[datasets_str]

    def show_list():
        return Datasets.DatasetsDic.keys()

