from subprocess import run
from os import listdir
from os.path import exists

from argparse import ArgumentParser

parser = ArgumentParser(description="""\
	Utility script for copying files used in imputation.
	Copies them to the desired equipments directory
""")

parser.add_argument('source_path', help="Source directory containing imputed equipments")

parser.add_argument('dest_path', help="Destination directory containing imputed equipments")

parser.add_argument('dest_tag', help="Type of classes used for the destiantion equipments, for example: pnct, sgp, hdm4.")
parser.add_argument('--source_tag', default='pnct', help="Type of classes used for the source equipments, for example: pnct, sgp, hdm4. Defaults to pnct.")

args, _ = parser.parse_known_args ()

source_path = args.source_path
assert exists(source_path), f'{source_path} directory does not exist'

dest_path = args.dest_path
assert exists(dest_path), f'{dest_path} directory does not exist'

print (source_path, dest_path)

for eq in listdir(f'{dest_path}'):
	outlis = f'{source_path}/{eq}/outli_eq_*.csv'
	imputs = f'{source_path}/{eq}/imput_eq_*.csv'
	lost_days = f'{source_path}/{eq}/lost_days_d_eq_*.csv'

	dest_folder = f'{dest_path}/{eq}'

	print (f'cp {outlis} {imputs} {lost_days} {dest_folder}')
	run (
		f'cp {outlis} {imputs} {lost_days} {dest_folder}',
		shell=True
	)

	for imput_eq in (
		a_csv 
		for a_csv in listdir(f'{dest_path}/{eq}') 
		if a_csv.startswith ('imput_eq_') and a_csv.endswith ('.csv')
	):
		source_imput = f'{dest_path}/{eq}/{imput_eq}'
		dest_imput = f'{dest_path}/{eq}/{imput_eq.replace (args.source_tag, args.dest_tag)}'
		
		print (f'mv {source_imput} {dest_imput}')
		run (
			f'mv {source_imput} {dest_imput}',
			shell=True
		)