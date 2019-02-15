import xmltodict
from tqdm import tqdm
import pandas as pd
import argparse


def get_aggregate_df(mxml_filename, groupby_time_frequency):
    print('Converting MXML to CSV', )
    df = pd.DataFrame(columns=['case_id', 'activity', 'resource', 'complete_timestamp'])

    with open(mxml_filename) as f:
        doc = xmltodict.parse(f.read())

    for instance in tqdm(doc['WorkflowLog']['Process']['ProcessInstance']):
        case_id = instance['@id']
        resource = None
        iterator = instance['AuditTrailEntry']
        iterator.sort(key=lambda x: x['Timestamp'])
        for entry in iterator:
            if entry['EventType'] == 'assign':
                try:
                    resource = entry['Originator']
                except KeyError:
                    resource = None
                continue
            df = df.append({
                'case_id': case_id,
                'activity': entry['WorkflowModelElement'],
                'resource': resource,
                'complete_timestamp': entry['Timestamp']
            }, ignore_index=True)

    print('Aggregating CSV Logs')
    df['completion_datetime'] = pd.to_datetime(df.complete_timestamp, format='%Y-%m-%dT%H:%M:%S.%f+00:00')
    df.drop(columns='complete_timestamp', inplace=True)
    df.fillna('None', inplace=True)

    aggregate_df = pd.DataFrame(columns=['timestamp', 'activity', 'resource', 'count'])

    grouped = df.groupby(pd.Grouper(key='completion_datetime', freq=groupby_time_frequency))
    for name1, group1 in tqdm(grouped):
        for name2, group2 in group1.groupby(['activity', 'resource']):
            aggregate_df = aggregate_df.append({
                'timestamp': name1,
                'activity': name2[0],
                'resource': name2[1],
                'count': len(group2['case_id'].unique())
            }, ignore_index=True)
    return aggregate_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate MXML logs')
    parser.add_argument('mxml_filename')
    parser.add_argument('save_filename')
    args = parser.parse_args()

    groupby_frequency = 'H'

    aggregate_data = get_aggregate_df(args.mxml_filename, groupby_frequency)
    aggregate_data.to_csv(args.save_filename, index=False)
    print('Written file to', args.save_filename)
