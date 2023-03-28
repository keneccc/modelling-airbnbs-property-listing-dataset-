import pandas as pd 
import csv


class data_prepearation():
    def __init__(self):
        self.df=pd.read_csv('clean_tabular_data.csv')
        


    def open_csv_file(self):
        with open('listing.csv', encoding="utf8") as f:
            airbnb_data=pd.read_csv(f)
            
        return(airbnb_data)

    def remove_rows_with_missing_ratings(self,df):
        clean_data_frame=df.dropna(subset=['Value_rating'])
        return(clean_data_frame)

    def combine_description_strings(self,df):
    
        df['Description'] = df['Description'].astype(str).str.replace(r'\[|\]|,', '',regex=False)
        df['Description'] = df['Description'].apply(lambda x: x.replace('About this space', ''))
        df['Description'] = df['Description'].apply(lambda x: x.replace("''", ''))
        df['Description'] = df['Description'].apply(lambda x: x.replace('"', ''))
        df['Description'] = df['Description'].apply(lambda x: x.replace("'", ''))
        
        df = df.dropna(subset=['Description'])
        return(df)

    def set_default_feature_values(self,df):
        df['guests'] = df['guests'].fillna(1)
        df['beds'] = df['beds'].fillna(1)
        df['bathrooms'] = df['bathrooms'].fillna(1)
        df['bedrooms'] = df['bedrooms'].fillna(1)
        return(df)


    def clean_tabular_data(self):
        df=self.remove_rows_with_missing_ratings(self.open_csv_file())
        df=self.combine_description_strings(df)
        self.set_default_feature_values(df)
        return df
        
    def load_airbnb(self,df,tg_column):
        
        features = df.drop(columns=['ID','Category','Title','Description','Amenities','Location','url','Unnamed: 19'], axis = 1)
        # Select the rows that have 'Somerford Keynes England United Kingdom'
        features = features[features['guests'] != 'Somerford Keynes England United Kingdom']
        labels = features[tg_column].values 
        feature = features.drop(labels=[tg_column], axis = 1)
     
        return(feature.values , labels) 
    



if __name__ == "__main__":
    
    prepare=data_prepearation()
    prepare.open_csv_file()
    df=prepare.clean_tabular_data()
    prepare.load_airbnb(df,"Price_Night")
    
    #df.to_csv('clean_tabular_data.csv', index=False)
    
