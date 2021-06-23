from subprocess import call
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Import or export the mysql database. You need to install 7-zip and mysql.')

parser.add_argument('database', type=str, help='A name of database')
parser.add_argument('user', type=str, help='Mysql user')
parser.add_argument('password', type=str, help='Mysql password')
parser.add_argument('-i', type=str, help='Import database', dest="import_database")
parser.add_argument('-o', help='Export database', action='store_true', dest="is_export")

args = parser.parse_args()

config = {
    'user': args.user,
    'password': args.password,
}

is_import = args.import_database is not None
is_export = args.is_export

config['database'] = args.database
config['file_name'] = '{}.7z'.format(config['database'])



if is_export:
    print('Start export the database:', config['database'])
    if config['file_name'] in os.listdir('out'):
        print('Delete out/{}'.format(config['file_name']))
        os.remove('out/{}'.format(config['file_name']))

    call('mysqldump --hex-blob --user={user} --password={password} {database} | 7z a -si out/{file_name}'.format(**config), shell=True)

if is_import:
    config['import_database'] = args.import_database
    print('Start import the database {database} to {import_database}:'.format(**config))
    if config['file_name'] in os.listdir('out'):
        print('Find', config['file_name'])

        call('7z x -so out/{file_name} | mysql --user={user} --password={password} {import_database}'.format(**config), shell=True)

    else:
        input('The file "out/{}" is not exist'.format(config['file_name']))