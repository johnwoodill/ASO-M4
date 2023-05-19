import pandas as pd
from urllib import request
import json


def parse_data_to_df(data):
    """
    Parse a dictionary and return a DataFrame with dateTime and value.
    """
    dateTime_values = []
    associated_values = []

    # Check if 'values' key exists
    if 'values' in data:
        # For each dictionary in the 'values' list
        for item in data['values']:
            # Check if 'value' key exists
            if 'value' in item:
                # For each dictionary in the 'value' list
                for sub_item in item['value']:
                    # Check if 'dateTime' key exists
                    if 'dateTime' in sub_item and 'value' in sub_item:
                        # Add 'dateTime' value to list
                        dateTime_values.append(sub_item['dateTime'])
                        # Add associated 'value' to list
                        associated_values.append(sub_item['value'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dateTime_values,
        'value': associated_values
    })
    
    return df






class StreamGauge:
    def __init__(self, site, start_date, end_date):
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        
    def get_json(self):
        # Get json file from USGS
        usgs_url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites="
                    f"{str(self.site)}&startDT={self.start_date}&endDT={self.end_date}&siteStatus=all")
        # Open json file
        with request.urlopen(usgs_url) as url:
            jdata = json.loads(url.read().decode())
        return jdata

    def proc_json(self):
        # Get StreamGauage info
        jdata = StreamGauge.get_json(self)
        
        # Loop through timeseries available from usgs
        outdat = pd.DataFrame()
        for i in range(len(jdata['value']['timeSeries'])):
            desc = jdata['value']['timeSeries'][i]['variable']['variableDescription']
            unit = jdata['value']['timeSeries'][i]['variable']['unit']['unitCode'].replace(" ", "_")
            stat = jdata['value']['timeSeries'][i]['variable']['options']['option'][0]['value']
            
            indat = parse_data_to_df(jdata['value']['timeSeries'][i])
            indat = indat.assign(usgs_site = self.site,
                                 var_desc = desc, 
                                 unit = unit,
                                 stat = stat)
            indat = indat.assign(date = pd.to_datetime(indat['date']).dt.strftime("%Y-%m-%d"))
            outdat = pd.concat([outdat, indat])
            
        return outdat



