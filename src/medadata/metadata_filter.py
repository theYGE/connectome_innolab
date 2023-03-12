import pandas as pd
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

class filter_metadata():
    """find out the patients/healthy participants in the metadata
    """
    def __init__(self, get_patients):
        """

        Args:
            get_patients (bool): True(False)- find out patients(healthy participants).
        """
        self.get_patients = get_patients
    
    def process_all_metadata(self):
        """go through all chunks of metadata and filter the targe participans.
        """
        self.read_download_list()
        self.get_metadata_chunks()
        self.load_field_dict()
        
        idx = 0
        for metadata in self.chunks:
            print(f"working on chunk {idx}, chunk size: {metadata.shape}")
            self.get_healthy_ones(metadata)
            idx += 1
    
    def load_field_dict(self):
        """load the disease information.
        """
        if self.get_patients:
            field_dict_name = "general_diseases.pickle"
        else:
            field_dict_name = "brain_diseases.pickle"
        with open(field_dict_name, 'rb') as f:
            fields_dict = pickle.load(f)
        self.disease_dict, self.feature_fields = fields_dict[0], fields_dict[1]
    
    def read_download_list(self):
        """read the available participants ID in our downloaded file list.
        """
        # filter the metadata by our downloaded biobank zip files
        zipfile = pd.read_excel("rs_ukb_downloads.xlsx", 
                        header=None)
        zipfile.rename(columns={0:"filename"}, inplace=True)
        zipfile_list = zipfile["filename"].apply(lambda x: str(x).split("_")[0])
        zipfile_list = zipfile["filename"].apply(lambda x: str(x).split("_")[0])
        self.zipfile_list = list(set(zipfile_list))
    
    def get_metadata_chunks(self):
        """load metadata in chunks.
        """
        # read metadata
        with open("target_col.pickle", 'rb') as f:
            target_col = pickle.load(f)
        target_col.append("eid")
        self.chunks = pd.read_csv("ukb_subset.csv", 
                            usecols=target_col,
                            on_bad_lines="skip",
                            skiprows=[103901],
                            chunksize=5000)
        
    def get_healthy_ones(self, metadata):
        """fileter the target participants of a single metadata chunk.

        Args:
            metadata (dataframe): the loaded metadata chunk.
        """
        metadata.set_index("eid", inplace=True)
        metadata.index = metadata.index.map(str)
        metadata.index.map(lambda x: x.replace(" ",""))
        metadata = metadata[metadata.index.isin(self.zipfile_list)]
        print(f"metadata size: {metadata.shape}")
        all_col = list(metadata.columns)
        metadata_concate = pd.DataFrame(index=metadata.index)
        metadata_concate["is_selected"] = True # if the participant will be selected
        target_metadata = pd.DataFrame()
        
        for field in self.disease_dict.keys():        
            if len(field) > 5: # for the fields share the same coding
                all_fields = field.split("_")
                target_fields = []
                for f in all_fields:
                    target_fields.extend([col for col in all_col if f in col])
            else:
                target_fields = [col for col in all_col if field in col]
                
            # concatenate the cols under the same field to a list, and drop NAN
            combined_str = metadata[target_fields].apply(
                            lambda x: ','.join(x.dropna().astype(str)),
                            axis=1)
            
            if self.get_patients:
                metadata_concate[field] = combined_str
                is_healthy = metadata_concate[field].apply(lambda x: "F" in x)
            else:
                metadata_concate[field] = combined_str.apply(lambda x: x.split(","))
                # we exclude the participant if he/she has any of the target diseases.
                is_healthy = ~metadata_concate[field].apply(lambda x: self.is_included(l1=x, l2=self.disease_dict[field]))
            metadata_concate["based_on_"+field] = is_healthy
            metadata_concate["is_selected"] = metadata_concate["is_selected"] & metadata_concate["based_on_"+field]
            
        for field in self.feature_fields.keys():
            field_re = re.compile(field+"\\-")
            target_fields = list(filter(field_re.match, all_col))
            # concatenate the cols under the same field to a list, and drop NAN
            combined_str = metadata[target_fields].apply(
                            lambda x: ','.join(x.dropna().astype(str)),
                            axis=1)

            metadata_concate[self.feature_fields[field]] = combined_str.apply(lambda x: x.split(","))
        
            metadata_concate[metadata_concate["is_selected"] == False]
            target_metadata = pd.concat([target_metadata, metadata_concate])
            
        if self.get_patients:
            target_metadata.to_csv("metadata_patients.csv")
        else:
            target_metadata.to_csv("metadata_healthy.csv")
    
    @staticmethod
    def is_included(l1, l2):
        """return True if any element of list l1 is included in list l2, else False.
        """
        # This function return True if any element of l1 is included in l2, else False
        for i in l1:
            if i in l2:
                return True
        return False 
    
if __name__ == "__main__":
    """get the healthy participants and patients
    """
    os.chdir(os.path.join(os.getcwd(), "data", "metadata"))
    filter_metadata(get_patients=True).process_all_metadata()
    filter_metadata(get_patients=False).process_all_metadata()

