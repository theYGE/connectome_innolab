import pandas as pd
import os
import re
import pickle

def generate_target_columns():
    """
    Find out the relavant/target columns in metadata, to reduce the memory needed during processing.
    The target columns will be saved in pickle file as a list.
    """
    metadata = pd.read_csv(os.path.join("ukb_subset.csv"))
    selected_col = []
    all_col = list(metadata.columns)
    # disease_fields = ['20002', '41203', '41205', '41202', '41204', '41270'] 
    # feature_fields = ["53","34", "31", "845", "6138"]
    # Features for modelling part:
    # 53: first day of attending 
    # 31: gender
    # 34: year of birth
    # 845-1.0	845-2.0: age completed full time education
    # 6138: QUALIFICATION

    target_fields = ["53","34", "31", "845", "6138", '20002', '41203', '41205', '41202', '41204', '41270']

    for field in target_fields:
        # target_cols = [(idx, col) for idx, col in enumerate(all_col) if field in col]
        field_re = re.compile(field+"\\-")
        # kept_cols = [col for col in (all_col) if field in col]
        kept_cols = list(filter(field_re.match, all_col)) 
        selected_col.extend(kept_cols)

    with open("target_col.pickle", 'wb') as f:
        pickle.dump(selected_col, f)

def get_general_diseases():
    """
    Get the filed ID and codings of general diseases, which will be used to define if a participant is healthy or not.
    The results will be saved in pickle file as a list of dictionaries.
    """
    # read the icd10 disease that we use to exclude the participants
    target = pd.read_csv("CNS_icd10_11_merge_full.csv", encoding= 'unicode_escape',
                        usecols=["icd10Code", "icd10Title"])
    target.dropna(axis=0, inplace=True)
    target_list = list(target["icd10Code"].apply(lambda x: x.replace(".", "")))
    target_block_list = [i for i in target_list if "-" in i] #block disease
    target_single_list = list(set(target_list) - set(target_block_list))
    
    # read metadata coding, and get the whole list of icd10 that UKbiobank used.
    data_coding = pd.read_csv("Codings.csv", 
                            encoding='latin-1')
    # data_dict = pd.read_csv(os.path.join(root_path, "data/metadata/Data_Dictionary_Showcase.csv"))

    data_coding = data_coding[data_coding["Coding"] == 19] # the coding of icd10 fields is 19
    icd_diseases = list(data_coding.Value)
    icd_diseases = [i.replace("Block ", "") if "Block" in i else i for i in icd_diseases]
    icd_block_list = [i for i in icd_diseases if "-" in i]
    icd_single_list = list(set(icd_diseases) - set(icd_block_list))

    # delete the icd entries that are not included in UKbiobank icd10.
    # single disease
    to_be_del = [i for i in target_single_list if i not in icd_single_list]
    target_single_list = list(set(target_single_list) - set(to_be_del))
    # block disease
    to_be_del = [i for i in target_block_list if i not in icd_block_list]
    target_block_list = list(set(target_block_list) - set(to_be_del))
    # double check
    assert len([i for i in target_single_list if i not in icd_single_list]) == 0
    assert len([i for i in target_block_list if i not in icd_block_list]) == 0
    
    # add all diseases start with "C: 'C00-C97', a special case. 
    to_be_added = [i for i in icd_block_list if "C" in i]
    target_block_list.extend(to_be_added)
    target_block_list = ["Block "+i for i in target_block_list]
    
    # combine the two in a list.
    full_target_list = []
    full_target_list.extend(target_block_list)
    full_target_list.extend(target_single_list)
    
    # still a few special case here:
    for i in ["I60", "I61", "I63", "I64", "G35", "G20", "F00", "F01", "F02", "F03", 
            "G30", "G31", "G32", "Block G30-G32", "G36", "G37"]:
        if i not in full_target_list:
            print(i)
            full_target_list.append(i)

    # from Boris' screenshot: Additionally these developmental disorders
    extra_disease = ["E700","E701","E720","E750","E752","E762","E763","E830","E720","F70","R620","F71","F819","F79","R625"
    "G318","F82","F840","F88","F849","R625","Q992","F89","F901","F802","F809","F952","R471"]
    full_target_list.extend(extra_disease)
    full_target_list = list(set(full_target_list))
    
    # the final disease dictionary!
    general_diseases = { 
    # FieldID: Coding
    "20002": ["1081.0", "1086.0", "1491.0", "1583.0", "1261.0", "1262.0", "1263.0", "1397.0"], 
    "41203_41205": ["430.0", "431.0", "434.0", "436.0", "340.0", "332.0", "290.0", "341.0"], 
    #  "41205": [430, 431, 434, 436, 340, 332, 290, 341],
    #  "41202": ["I60", "I61", "I63", "I64", "G35", "G20", "F00", "F01", "F02", "F03", 
    #            "G30", "G31", "G32", "Block G30-G32", "G36", "G37"], 
    #  "41204": ["I60", "I61", "I63", "I64", "G35", "G20", "F00", "F01", "F02", "F03", 
    #            "G30", "G31", "G32", "Block G30-G32", "G36", "G37"], 
    '41202_41204_41270': full_target_list, 
    }

    feature_fields = {"53": "sample_day","34": "birth_year", "31": "gender", 
                    "845": "age_complete_edu", "6138": "qualification"}
    
    with open("general_diseases.pickle", 'wb') as f:
        pickle.dump([general_diseases, feature_fields], f)

def get_brain_diseases():
    """
    Save the filed ID and codings of brain diseases, which will be used to define if a participant has brain disease or not.
    The results will be saved in pickle file as a list of dictionaries.
    """
    disease_dict = { 
    # FieldID: Coding
    '41202_41204_41270': ["F00", "F000", "F001", "F002", "F009", 
                        "G30", "G300", "G301", "G308", "G309", ], 
    }

    feature_fields = {"53": "sample_day","34": "birth_year", "31": "gender", 
                    "845": "age_complete_edu", "6138": "qualification"}
    with open("brain_diseases.pickle", 'wb') as f:
        pickle.dump([disease_dict, feature_fields], f)

if __name__ == "__main__":
    os.chdir(os.path.join(os.getcwd(), "data", "metadata"))
    generate_target_columns()
    get_general_diseases()
    get_brain_diseases()