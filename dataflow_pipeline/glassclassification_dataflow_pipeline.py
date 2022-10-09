import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import argparse

def run(argv=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--projectid',dest='projectid')
    parser.add_argument('--region',dest='region')
    parser.add_argument('--rootbucket',dest='rootbucket')
    parser.add_argument('--bqdataset',dest='bqdataset')
    parser.add_argument('--bqtable',dest='bqtable')
    known_args, _ = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project=known_args.projectid,
        region=known_args.region,
        temp_location= known_args.rootbucket + '/dataflow/temp')

    with beam.Pipeline(options=pipeline_options) as p:
        
        def dropNull(data):
            return not None in data.values()

        def convertTypes(data):
            try:
                data['ID'] = int(data['ID']) if 'ID' in data else None
                data['RI'] = float(data['RI']) if 'RI' in data else None
                data['Na'] = float(data['Na']) if 'Na' in data else None
                data['Mg'] = float(data['Mg']) if 'Mg' in data else None
                data['Al'] = float(data['Al']) if 'Al' in data else None
                data['Si'] = float(data['Si']) if 'Si' in data else None
                data['K'] = float(data['K']) if 'K' in data else None
                data['Ca'] = float(data['Ca']) if 'Ca' in data else None
                data['Ba'] = float(data['Ba']) if 'Ba' in data else None
                data['Fe'] = float(data['Fe']) if 'Fe' in data else None
                data['Type'] = int(data['Type']) if 'Type' in data else None
            except:
                data = dict.fromkeys(data, None)
            return data

        def selectColumns(data):
            del data['ID']
            return data
        
        pipe = (p
            | 'ReadDataFromGC' >> beam.io.ReadFromText(known_args.rootbucket + '/glass.csv', skip_header_lines=1)
            | 'SplitData' >> beam.Map(lambda x: x.split(','))
            | 'ToDict' >> beam.Map(lambda x: {
                'ID': x[0],
                'RI': x[1],
                'Na': x[2],
                'Mg': x[3],
                'Al': x[4],
                'Si': x[5],
                'K': x[6],
                'Ca': x[7],
                'Ba': x[8],
                'Fe': x[9],
                'Type': x[10]})
            | 'ConvertTypes' >> beam.Map(convertTypes)
            | 'DropNull' >> beam.Filter(dropNull)
            | 'SelectColumns' >> beam.Map(selectColumns)
            | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                known_args.projectid + ':' + known_args.bqdataset + '.' + known_args.bqtable,
                schema='RI:FLOAT,Na:FLOAT,Mg:FLOAT,Al:FLOAT,Si:FLOAT,K:FLOAT,Ca:FLOAT,Ba:FLOAT,Fe:FLOAT,Type:INTEGER',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                custom_gcs_temp_location = known_args.rootbucket + '/bigquery/temp')
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
