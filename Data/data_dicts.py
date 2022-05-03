character_dict = {
    'Barney':{
        'df_filename': 'Barney.csv',
        'prediction_filename': 'barney_prediction',
        'checkpoint_folder': 'barney_model',
        'classifier_name': 'barney_classifier',
        'classifier_df': 'barney_classifier.csv',
        'encoded_lines_filename': 'barney_encoded_lines.npy',
        'source': 'HIMYM',
        'delete_names':["Barney's Secretary",
                        'Marshall to Barney',
                        "Barney's mom",
                        'Ted, from seeing Barney',
                        'Lily, holding Barney',
                        'Marshall, on the phone with Barney',
                        "At Casa a pezzi. Barney is playing the piano.Ted's father",
                        'Marshall, to the girl Barney is talking to']
    },
    'Sheldon':{
        'df_filename': 'Sheldon.csv',
        'prediction_filename': 'sheldon_prediction',
        'checkpoint_folder': 'sheldon_model',
        'classifier_name': 'sheldon_classifier',
        'classifier_df': 'sheldon_classifier.csv',
        'encoded_lines_filename': 'sheldon_encoded_lines.npy',
        'source': 'TBBT',
        'delete_names':[]
    },
    'Harry':{
        'df_filename': 'Harry.csv',
        'prediction_filename': 'harry_prediction',
        'checkpoint_folder': 'harry_model',
        'classifier_name': 'harry_classifier',
        'classifier_df': 'harry_classifier.csv',
        'encoded_lines_filename': 'harry_encoded_lines.npy',
        'source': 'HP',
        'delete_names':[]
    },
    'Fry':{
        'df_filename': 'Fry.csv',
        'prediction_filename': 'fry_prediction',
        'checkpoint_folder': 'fry_model',
        'classifier_name': 'fry_classifier',
        'classifier_df': 'fry_classifier.csv',
        'encoded_lines_filename': 'fry_encoded_lines.npy',
        'source': 'Futurama',
        'delete_names':['Mrs fry',
                        'Mr fry',
                        'Luck of the fryrish']
    },
    'Bender':{
        'df_filename': 'Bender.csv',
        'prediction_filename': 'bender_prediction',
        'checkpoint_folder': 'bender_model',
        'classifier_name': 'bender_classifier',
        'classifier_df': 'bender_classifier.csv',
        'encoded_lines_filename': 'bender_encoded_lines.npy',
        'source': 'Futurama',
        'delete_names':[]
    },
    'Vader':{
        'df_filename': 'Vader.csv',
        'prediction_filename': 'vader_prediction',
        'checkpoint_folder': 'vader_model',
        'classifier_name': 'vader_classifier',
        'classifier_df': 'vader_classifier.csv',
        'encoded_lines_filename': 'vader_encoded_lines.npy',
        'source': 'SW',
        'delete_names':["INT. DARTH VADER'S WINGMAN - COCKPIT"]
    },
    'Joey': {
        'df_filename': 'Joey.csv',
        'prediction_filename': 'joey_prediction',
        'checkpoint_folder': 'joey_model',
        'classifier_name': 'joey_classifier',
        'classifier_df': 'joey_classifier.csv',
        'encoded_lines_filename': 'joey_encoded_lines.npy',
        'source':'Friends',
        'delete_names':["Joeys Sisters",
                        'Joey\'s Date', 
                        "Joey's Look-A-Like", 
                        'Joeys Sister', 
                        "Joey's Doctor", 
                        "Joey's Hand Twin", 
                        'Joeys Date', 
                        'Joeys Grandmother']
    },
    'Phoebe': {
        'df_filename': 'Phoebe.csv',
        'prediction_filename': 'phoebe_prediction',
        'checkpoint_folder': 'phoebe_model',
        'classifier_name': 'phoebe_classifier',
        'classifier_df': 'phoebe_classifier.csv',
        'encoded_lines_filename': 'phoebe_encoded_lines.npy',
        'source':'Friends',
        'delete_names':['Amy turns around to Phoebe',
                        'Phoebe Waitress']
    },
    'Default': None
}

source_dict = {
    'HIMYM':{
        'dataset_folder': 'Episodes',
        'df_filename': 'HIMYM.csv'
    },
    'Futurama':{
        'dataset_folder': 'Episodes',
        'df_filename': 'Futurama.csv'
    },
    'Friends':{
        'dataset_folder': None,
        'df_filename': 'Friends.csv'
    },
    'HP':{
        'dataset_folder': None,
        'df_filename': 'HP.csv'
    },
    'SW':{
        'dataset_folder': 'Scripts',
        'df_filename': 'SW.csv'
    },
    'TBBT':{
        'dataset_folder': 'Episodes',
        'df_filename': 'TBBT.csv'
    },
}

random_state = 31239812