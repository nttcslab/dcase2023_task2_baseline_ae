import argparse
import csv
import os
import pathlib
if __name__ == "__main__":
    from plot_common import Figdata, show_figs
else:
    from tools.plot_common import Figdata, show_figs

def csv_to_figdata(file_path, column_heading_list=["loss"], xlabel="epoch", ylabel="loss", fig_count=1, cut_first_epoch=False, is_display_console=False):
    figdata_list = []
    for i in range(fig_count):
        data_dict = {}
        column_heading_list_temp = column_heading_list[i] if fig_count > 1 else column_heading_list
        for column_heading in column_heading_list_temp:
            data_dict[column_heading] = []

        with open(file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for column_heading in column_heading_list_temp:
                    data_dict[column_heading].append(float(row[column_heading]))
        data_list = list(data_dict.values())
        if cut_first_epoch:
            data_list = [ x[1:] for x in data_list ]
        if len(data_dict.keys()) >= 2:
            figdata_list.append(Figdata(
                data=data_list[0],
                data2=data_list[1:],
                type="plot",
                labels=column_heading_list_temp,
                xlabel=xlabel,
                ylabel=ylabel
            ))
        else:
            figdata_list.append(Figdata(
                data=data_list[0],
                type="plot",
                labels=column_heading_list_temp,
                xlabel=xlabel,
                ylabel=ylabel
            ))

    export_path =os.path.join(os.path.dirname(file_path), pathlib.PurePath(file_path).stem + ".png")
    show_figs(
        *figdata_list,
        sup_title=os.path.splitext(os.path.basename(file_path))[0],
        fold_interval=fig_count,
        export_path=export_path,
        is_display_console=is_display_console
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("--file_path", type=str, default='../stable/', metavar='N')
    parser.add_argument("--column_heading", type=str, nargs='*', default=["loss"], metavar='N')
    parser.add_argument("--xlabel", type=str, default=["epoch"], metavar='N')
    parser.add_argument("--ylabel", type=str, default=["loss"], metavar='N')
    args = parser.parse_args()

    csv_to_figdata(
        file_path=args.file_path,
        column_heading_list=args.column_heading,
        xlabel=args.xlabel,
        ylabel=args.ylabel
    )
