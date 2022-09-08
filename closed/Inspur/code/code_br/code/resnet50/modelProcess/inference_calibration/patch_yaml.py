import re, argparse #, yaml

def set_yaml_field(yaml_file, data_path=None, im_list_path=None):
    data = ""
    with open(yaml_file, "r", encoding="utf-8") as f:
        # data = yaml.load(stream=f, Loader=yaml.FullLoader)
        # data["quantization"]["calibration"]["dataloader"]["dataset"]["ImagenetRaw"]["data_path"] = data_path
        # data["quantization"]["calibration"]["dataloader"]["dataset"]["ImagenetRaw"]["image_list"] = im_list_path

        # data["evaluation"]["accuracy"]["dataloader"]["dataset"]["ImagenetRaw"]["data_path"] = data_path
        # data["evaluation"]["accuracy"]["dataloader"]["dataset"]["ImagenetRaw"]["image_list"] = im_list_path

        # data["evaluation"]["performance"]["dataloader"]["dataset"]["ImagenetRaw"]["data_path"] = data_path
        # data["evaluation"]["performance"]["dataloader"]["dataset"]["ImagenetRaw"]["image_list"] = im_list_path
        for line in f:
            if "data_path: " in line:
                # line.insert(21, data_path + " # ")
                line = line[:21] + data_path + "\n"
            if "image_list: " in line:
                # line.insert(22, im_list_path + " # ")
                line = line[:22] + im_list_path + "\n"
            data += line

    with open(yaml_file, "w", encoding="utf-8") as f:
        # yaml.dump(data=data, stream=ff, allow_unicode=True)
        f.write(data)

def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--yaml_path", help="yaml path")
    args_parser.add_argument("--data_path", help="dataset path")
    args_parser.add_argument("--image_list", help="image list text file")
    args = args_parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    # print(args.data_path)
    # print(args.image_list)
    set_yaml_field(args.yaml_path, args.data_path, args.image_list)     
