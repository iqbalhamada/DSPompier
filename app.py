from collections import OrderedDict
import streamlit as st
from PIL import Image
#from prepro import import_data
import pandas as pd
import numpy as np
import datetime
import shap
from tqdm import tqdm
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns

#import geopandas as gpd
from geopy import distance
from pyproj import Proj, transform,Transformer

import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
import tensorflow as tf
from tensorflow.keras import callbacks
print(tf.__version__)
print(sklearn.__version__)
print(tf.config.list_physical_devices('GPU'))

import os
from tqdm._tqdm_notebook import tqdm_notebook
import warnings 
warnings.filterwarnings('ignore')
tqdm_notebook.pandas()


# Charger les images
intro_image = Image.open("..\Images\LFBImage.JPEG")   
sidebar_image = Image.open("..\Images\LFBImage1.JPG")  

#intro_image = intro_image.resize((100, 100))
sidebar_image = sidebar_image.resize((180, 180))

# Informations de l'équipe
team_members = [
    {
        "name": "Nom Prénom 1",
        "linkedin": "https://www.linkedin.com/in/username1",
        "github": "https://github.com/username1"
    },
    {
        "name": "Nom Prénom 2",
        "linkedin": "https://www.linkedin.com/in/username2",
        "github": "https://github.com/username2"
    },
    {
        "name": "Nom Prénom 3",
        "linkedin": "https://www.linkedin.com/in/username3",
        "github": "https://github.com/username3"
    },
    {
        "name": "Nom Prénom 4",
        "linkedin": "https://www.linkedin.com/in/username4",
        "github": "https://github.com/username4"
    }
]

# Définir les sections de la sidebar
sections = [
    "Présentation du projet",
    "Description des datasets",
    "DataVizualisation",
    "Série temporelle",
    "Machine Learning",
    "Deep Learning",
    "Perspectives"
]

# Afficher la sidebar
st.sidebar.title("Menu")
st.sidebar.image(sidebar_image, use_column_width=True)
section = st.sidebar.radio("Aller à", sections)

#st.sidebar.markdown("---")
st.sidebar.header("Equipe")

for member in team_members:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="margin-right: 10px;"><strong>{member['name']}</strong></span>
            <a href="{member['linkedin']}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" alt="LinkedIn" width="60"></a>
            <a href="{member['github']}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" width="20"></a>
        </div>
        """, unsafe_allow_html=True
    )

# Présentation du projet 
if section == "Présentation du projet":
    st.image(intro_image, use_column_width=True)
    st.title("Introduction")
    
    # Titre de l'application
    #st.title('Analyse et Estimation des Temps de Réponse des Pompiers de Londres')

    # Paragraphe donné
    st.markdown("""
    L’objectif de ce projet est d’analyser et/ou d’estimer les temps de réponse et de mobilisation de la Brigade des Pompiers de Londres.\n
    La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde.\n
    Le premier jeu de données fourni contient les détails de chaque incident traité depuis janvier 2009. Des informations sont fournies sur la date et le lieu de l'incident ainsi que sur le type d'incident traité.\n
    Le second jeu de données contient les détails de chaque camion de pompiers envoyé sur les lieux d'un incident depuis janvier 2009. Des informations sont fournies sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident.\n
    Notre projet se déroule en plusieurs temps :
    - Analyse des jeux de données, datavisualisation et prép-processing
    - Modélisation au travers de différentes méthodes de Machine Learning\n
    Ceci afin de pouvoir définir un modèle prédictif le plus fiable possible pouvant être exploité par les professionnels concernés.
    """)

# Description des datasets
elif section == "Description des datasets":
    #st.title("Description des datasets")
    st.write("Dans cette section, nous allons décrire les données.")
    # Page "exploration Datasets"

    # Import packages nécessaires
    import pandas as pd
    import streamlit as st

    st.markdown("<h1 style='text-align: center; color: blue;'>Description des datasets</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; color: black;'>Source des jeux de données</h2>", unsafe_allow_html=True)

    st.write("Dans le cadre de ce projet, nous disposons de deux jeux de données, issus du site **London Store**, portail de partage de données relatives à la capitale, gratuit et ouvert à tous*.")

    st.write("- Un premier jeu de données (**Incidents**) contient les détails de chaque incident traité par la brigade de Londres entre 2018 et 2023.")

    if st.checkbox("_Afficher la structure de la table **Incidents**_") :
        st.write("- 39 colonnes et 659 611 observations")
        st.write("- Clé primaire : IncidentNumber")
        
    if st.checkbox("_Afficher valeurs manquantes de la table **Incidents**_") :
        st.image(r'..\Images\valeurs_manquantes_df_incidents.jpg', caption="Variables à plus 10% de valeurs manquantes", width=600)

    st.write("- Un second jeu de données (**Mobilisations**) contient les détails de chaque ressource (camion de pompiers) envoyé sur les lieux d’un incident entre 2015 et 2023.")         

    if st.checkbox("_Afficher la structure de la table **Mobilisations**_") :
        st.write("- 22 colonnes et 492 088 observations")
        st.write("- Clé primaire : ResourceMobilisationId")
        
    if st.checkbox("_Afficher valeurs manquantes de la table **Mobilisations**_") :
        st.image(r'..\Images\valeurs_manquantes_df_mobilisations.jpg', caption="Variables à plus 10% de valeurs manquantes", width=600)

    st.write("")

    st.markdown("<h2 style='text-align: left; color: black;'>Description de la base de travail</h2>", unsafe_allow_html=True)

    st.write("Après les étapes de preprocessing, nous disposons d'une base de travail unique, comportant les données d'incidents et de ressources mobilisées, avec 6 ans de profondeur d'historique.")
    try:
        # Tente de lire le fichier CSV localement
        df = pd.read_csv(r'..\data\LFB_data_preprocess.csv')

    except FileNotFoundError:
    # Si le fichier local n'est pas trouvé, charge depuis Dropbox
        df = pd.read_csv(r'https://www.dropbox.com/scl/fi/61tb0gn3gvl1o4lv8dg81/LFB_data_preprocess.csv?rlkey=8cpwwtzpk8patb83nedcwn9hb&st=vyvg0x41&dl=1')
    
    #df = pd.read_csv(r'..\data\LFB_data_preprocess.csv')
    
    st.write("**- Extrait du dataframe retraité :**")
    st.dataframe(df.head())

    st.write("**- Dimensions du dataframe retraité :**")
    st.write(df.shape)

    st.write("")

    st.markdown("<h10 style='text-align: left; color: black;'>*_Sources des données :_</h10>", unsafe_allow_html=True)
    st.write("https://data.london.gov.uk/dataset/london-fire-brigade-incident-records")
    st.write("https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records")
    # Ajoutez ici votre code de visualisation de données
# DataVizualisation
elif section == "DataVizualisation":
    #st.title("DataVizualisation")
    #st.write("Dans cette section, nous allons visualiser les données.")
    # Ajoutez ici votre code de visualisation de données
    import streamlit as st
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    from PIL import Image

    st.markdown("<h1 style='text-align: center; color: blue;'>Datavisualisation</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>Variable cible</h2>", unsafe_allow_html=True)

    st.image(r"..\Images\Distribution_variable_cible.jpg", caption="Distribution des temps d'attente (variable cible)", width=700)

    st.image(r'../Images/Distribution_gamma_variable_cible.jpg', caption=" La variable cible ne suit pas une distribution Gamma malgré les apparences", width=700)

    st.image(r'../Images/Distribution_variable_cible_hue.jpg', caption="Distribution des temps d'attente par type d'incident (variable cible)", width=700)



    st.markdown("<h2 style='text-align: center; color: black;'>Types d'interventions</h2>", unsafe_allow_html=True)

    st.image(r'../Images/Boxplot_temps_attente_type_intervention.jpg', caption="Temps d'attente par type d'intervention", width=1400)

    st.image(r'../Images/Mobilisations_Description_Pie.jpg', caption="Description des différents types de mobilisation", width=1000)

    st.image(r'../Images/Mobilisations_Incidents.jpg', caption="Nombre de mobilisations par incidents", width=600)



    st.markdown("<h2 style='text-align: center; color: black;'>Evolution temporelle</h2>", unsafe_allow_html=True)

    st.image(r'../Images/Nb_mobilisations_mensuelles_diachronique.jpg', caption="Evolution mensuelle du nombre d'appels", width=1300)

    st.image(r'../Images/Distribution_nb_appels_heure.jpg', caption="Distribution du nombre d'appels par heure", width=1500)

    st.image(r'../Images/Distribution_nb_appels_jour_mois.jpg', caption="Distribution du nombre d'appels par jour et par mois", width=1500)

    st.image(r'../Images/Distribution_nb_appels_an_moti.jpg', caption="Distribution du nombre d'appels par année et par motif", width=1100)



    st.markdown("<h2 style='text-align: center; color: black;'>Représentations spatiales</h2>", unsafe_allow_html=True)

    #background = Image.open('Images/Carte_Londres.jpg')
    #col1, col2, col3 = st.columns([0.2, 150, 0.2])
    #col2.image(background, use_column_width=True)

    st.image(r'..\Images\Carte_Londres.jpg', caption='Carte de Londres avec, en rouge, les casernes les plus sollicitées et en vert les casernes les moins sollicitées', width=1500)

    st.image(r'../Images/Carte_Londres_distances_casernes.jpg', caption="Carte de Londres avec, en rouge, les casernes avec les distances parcourues les plus longues et en vert les casernes les distances parcourues les moins longues", width=1500)

    st.image(r'../Images/Carte_Londres_attendance_time_casernes.jpg', caption="Carte de Londres avec, en rouge, les casernes avec les temps d'attente les plus long et en vert les casernes les moins longs", width=1500)

    



# Série temporelle
elif section == "Série temporelle":
    st.title("Série temporelle")
    st.write("Dans cette section, nous allons étudier la Série temporelle.")
    st.markdown("<h1 style='text-align: center; color: blue;'>Etude temporelle des appels</h1>", unsafe_allow_html=True)

    paragraphe1 = '''Pour pouvoir maintenir un niveau de performance efficace des pompiers de Londres, pouvoir anticiper une évolution probabliste du nombre d'appels 
    est un paramètre important. Pouvoir anticiper des phénomènes saisonniers ou une croissance du nombre d'appels pourrait permettre une adaptation des moyens
    (tant en ressources huamines que logistiques).
    En ce sens, nous avons réalisé une étude temporelle sur le nombre d'appels reçus par mois pour pouvoir prédire l'évolution potentielle sur les 2 années à venir.
    '''
    st.markdown(paragraphe1)

    st.markdown("<h2 style='text-align: left; color: black;'>Nombre d'appels réceptionnés par les pompiers de Londres</h2>", unsafe_allow_html=True)
    image_path = r"..\Images\Nombredappel.jpg"
    image_1 = Image.open(image_path)
    # Afficher l'image avec Streamlit
    st.image(image_1, caption="Nombre d'appels réceptionnés", width=700)
    #st.image(, caption="", width=700)

    st.image(r'..\Images\comparaison.jpg', caption="Comparaison modèle additif et modèle multiplicatif", width=700)

    paragraphe2 = '''Cette comparaison de modèle permet de conclure de façon claire que l'évolution du nombre d'appels correspond à un modèle
    multiplicatif. Cette information est extrêment importante à intégrer car cela signifie qu'au fur et à mesure des années, ce nombre
    d'appels sera de plus en plus important et non sur un acroissement constant.
    De plus, nous pouvons identifier des phénomènes de saisonnalité (de façon annuelle) avec des pics d'afflux en période de vacances notamment
    '''
    st.markdown(paragraphe2)

    st.markdown("<h2 style='text-align: left; color: black;'>Prévision d'appels</h2>", unsafe_allow_html=True)
    paragraphe3 = '''le modèle prédictif avec la méthode "SARIMA" nous permet de pouvoir prédire l'évolution du nombre d'appels sur les 2 années à venir, de la 
    façon suivante:
    '''
    st.markdown(paragraphe3)

    st.image(r'..\Images\Prevision.jpg', caption="Prédiction d'évolution des appels", width=700)

    paragraphe4 = '''Le tracé en pointillé correspond à l'évolution moyenne prédite. La plage grisée correspond à la plage d'erreur (écart-type) potentielle.
    Aini, plus on avance dans le temps de prévision, plus de façon logique la plage d'erreur augmente (liée aux phénomènes d'incertitudes d'évènements)
    Il sera important dans la continuité de vie du projet que ce modèle puisse être ré-interrogé pour:
    - identifier la différence entre la prédiction et la réalité du nombre d'appels reçu en 2025
    - optimiser le modèle de prédiction pour fiabiliser cette donnée qui a pour but d'anticiper les moyens à allouer
    '''
    st.markdown(paragraphe4)


# Machine Learning
elif section == "Machine Learning":
    try:
        # Tente de lire le fichier CSV localement
        df = pd.read_csv(r'..\data\LFB_data_preprocess.csv')

    except FileNotFoundError:
    # Si le fichier local n'est pas trouvé, charge depuis Dropbox
        df = pd.read_csv(r'https://www.dropbox.com/scl/fi/61tb0gn3gvl1o4lv8dg81/LFB_data_preprocess.csv?rlkey=8cpwwtzpk8patb83nedcwn9hb&st=vyvg0x41&dl=1')
    #df = df.dropna(subset = 'duration')
    #df = df.drop('duration', axis = 1)
    print(df.shape)
    #nb_samples = 300000
    #df = df.sample(nb_samples)
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['year'] = df['date'].dt.year
    df['wd'] = df['date'].dt.weekday
    df = df.drop({'station_lat', 'station_lon', 'mob_coords', 'station_coords','IncidentNumber', 'date','USRN','PropertyType',
                'IncGeo_WardNameNew','Postcode_district','IncGeo_BoroughName', 'IncidentStationGround'}, axis = 1).set_index('ResourceMobilisationId')
    # 'Pump Order' et 'NumStationsWithPumpsAttending': Catégorielle ?
    df = df.drop(['Resource_Code',
              'DeployedFromStation_Code',
              'PerformanceReporting', # Connue ex post
              'DateAndTimeMobilised', # On a les deltas temporels
              'DateAndTimeMobile', # On a les deltas temporels
              'DateAndTimeArrived', # On a les deltas temporels
              'TurnoutTimeSeconds', # Décomposition de la varaible cible
              'TravelTimeSeconds', # Décomposition de la variable cible
              'UPRN', # ID de propriété ; trop de valeurs uniques pour être pertiente
              'DateAndTimeLeft', # Non pertinent pour la variable cibre
              'PlusCode_Code',  # Une seule modalité
              'PlusCode_Description', # Une seule modalité
              'AddressQualifier', # Connue ex post
              'IncGeo_BoroughCode', # Redondant avec le 'IncGeo_BoroughName'
              'ProperCase', # Redondant avec le 'IncGeo_BoroughName'
              'IncGeo_WardCode', # Redondant avec le 'IncGeo_WardNameNew'
              'IncGeo_WardName', # Redondant avec le 'IncGeo_WardNameNew'
              'FRS', # Une seule modalité
              'PumpMinutesRounded', # Connue ex post
              'Notional Cost (£)',  # Connu ex post
              'FirstPumpArriving_AttendanceTime', # Connue ex post
              'CalYear_x',
              'HourOfCall_x',     
              'CalYear_y',
              'HourOfCall_y',                           
              'TimeOfCall',
              'mois',
              'jour',
              'annee'
             ], axis = 1)
    st.dataframe(df.head())
        #col = 'IncidentStationGround'
    cols_cat = ['DeployedFromStation_Name','DeployedFromLocation','IncidentGroup','StopCodeDescription',
                'PropertyCategory','FirstPumpArriving_DeployedFromStation']
    #print("Nb de modalités : ",len(df[col].unique()))
    #print(df[col].value_counts())
    #print(df[col].unique())

    y = df['AttendanceTimeSeconds']
    cols_num = ['distance','NumCalls','NumStationsWithPumpsAttending','Northing_rounded',
                'Easting_rounded','NumPumpsAttending','PumpCount','PumpOrder','hour','year','wd']
    #cols_num
    df_cat_dum = pd.get_dummies(df[cols_cat].astype(str))
    #print(df_cat_dum.shape)
    X = pd.concat([df_cat_dum,df[cols_num]], axis = 1)
    #print(X.shape)
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_train_pred = model.predict(X_train)
    # mae_train = mean_absolute_error(y_train, y_train_pred)
    # #y_test_pred = model.predict(X_test)
    # mae_test = mean_absolute_error(y_test, y_test_pred)

    #print("Mean Absolute Error (Train):", mae_train)
    #print("Mean Absolute Error (Test):", mae_test)
    # joblib.dump(model, "model_reg_line")
    ################################################################
  
  ################################################################
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # reg_KNN = KNeighborsRegressor()
    # reg_KNN.fit(X_train, y_train)
    
    # y_pred_test_KNN = reg_KNN.predict(X_test)
    # y_pred_train = reg_KNN.predict(X_train)
    # #print("MAE Train: ", round(mean_absolute_error(y_train, y_pred_train),1))
    # #print("MAE Test: ", round(mean_absolute_error(y_test, y_pred_test),1))
    # joblib.dump(reg_KNN, "model_reg_KNN")

    ########################################################################
    ########################################################################

    # reg_XGB = XGBRegressor()
    # reg_XGB.fit(X_train, y_train)
    # #y_pred_test_XGB = reg_XGB.predict(X_test)
    # y_pred_train = reg_XGB.predict(X_train)
    # #print("MAE Train: ", round(mean_absolute_error(y_train, y_pred_train),1))
    # #print("MAE Test: ", round(mean_absolute_error(y_test, y_pred_test),1))
    # joblib.dump(reg_XGB, "Model_reg_XGB")
    ########################################################################
    ########################################################################
    # model_grid = XGBRegressor()
    # param_grid = {'eta' : [0.1,0.2,0.3], 'max_depth' : [12,13,14]}
    # #param_grid = {'eta' : [0.3], 'max_depth' : [6]}
    # grid_cv = GridSearchCV(estimator = model_grid,
    #                     param_grid = param_grid,
    #                     cv = 5)
    # grid_cv.fit(X_train, y_train)
    # #y_pred_test_GXBR = grid_cv.predict(X_test)
    # y_pred_train = grid_cv.predict(X_train)
    # #print('Meilleurs paramètres : ',grid_cv.best_params_)
    # #print("MAE Train: ", round(mean_absolute_error(y_train, y_pred_train),1))
    # #print("MAE Test: ", round(mean_absolute_error(y_test, y_pred_test),1))
    # joblib.dump(grid_cv, "Model_model_grid")
    import matplotlib.pyplot as plt
    try:
        # Tente de lire le fichier CSV localement
        df = pd.read_csv(r'..\data\LFB_data_preprocess.csv')

    except FileNotFoundError:
    # Si le fichier local n'est pas trouvé, charge depuis Dropbox
        df = pd.read_csv(r'https://www.dropbox.com/scl/fi/61tb0gn3gvl1o4lv8dg81/LFB_data_preprocess.csv?rlkey=8cpwwtzpk8patb83nedcwn9hb&st=vyvg0x41&dl=1')
    
    #data = pd.read_csv(r'..\data\LFB_data_preprocess.csv')
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    # Titre de l'application
    st.title("Estimation du Temps Moyen d'Intervention")

    # Widget de sélection pour la distance
    distance = st.slider("Sélectionnez la distance (en km):", 0, 25, 1)

    # Widget de sélection pour le type d'incident
    incident_type = st.selectbox("Sélectionnez le type d'incident:", data['StopCodeDescription'].unique())

    # Widget de sélection pour la station de départ
    station = st.selectbox("Sélectionnez la station de départ:", data['DeployedFromStation_Name'].unique())

    # Filtrer les données en fonction des sélections
    filtered_data = data[(data['distance'] <= distance) & 
                        (data['StopCodeDescription'] == incident_type) & 
                        (data['DeployedFromStation_Name'] == station)]

    # Afficher les données filtrées pour vérification
    st.write("Données Filtrées:")
    st.dataframe(filtered_data)

    # Afficher des statistiques descriptives des données filtrées
    st.write("Statistiques Descriptives des Données Filtrées:")
    st.write(filtered_data['AttendanceTimeSeconds'].describe())

    # Calculer le temps moyen d'intervention
    if not filtered_data.empty:
        mean_time = filtered_data['AttendanceTimeSeconds'].mean()
        mean_minutes = int(mean_time // 60)
        mean_seconds = int(mean_time % 60)
        st.write(f"Le temps moyen d'intervention est de {mean_minutes} minutes et {mean_seconds} secondes.")
    else:
        st.write("Aucune donnée ne correspond aux critères sélectionnés.")

    # Afficher un graphique de distribution du temps d'intervention
    st.write("Distribution du Temps d'Intervention pour les Critères Sélectionnés:")
    if not filtered_data.empty:
        fig, ax = plt.subplots()
        ax.hist(filtered_data['AttendanceTimeSeconds'] / 60, bins=30)
        ax.set_title("Distribution du Temps d'Intervention (en minutes)")
        ax.set_xlabel("Temps d'Intervention (minutes)")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
    else:
        st.write("Aucune donnée à afficher.")

    model_DecisionTree_regressor = joblib.load("..\models\model_DecisionTree_regressor")
    model_gb_regressor = joblib.load("..\models\model_gb_regressor")
    model_Grid_CV = joblib.load("..\models\model_Grid_CV")
    model_reg_line = joblib.load("..\models\model_reg_line")
    model_Rf_reg = joblib.load("..\models\model_Rf_reg")  # fichier volumineux !!! 6GB
    model_XGB_reg = joblib.load("..\models\model_XGB_reg")
    model_KNN_reg = joblib.load("..\models\model_KNN_reg")


    y_pred_model_DecisionTree_regressor= model_DecisionTree_regressor.predict(X_test)
    y_pred_model_gb_regressor =model_gb_regressor.predict(X_test)
    y_pred_model_Grid_CV=model_Grid_CV.predict(X_test) 
    y_pred_model_reg_line=model_reg_line.predict(X_test)
    #y_pred_model_Rf_reg=model_Rf_reg.predict(X_test) 
    y_pred_model_XGB_reg=model_XGB_reg.predict(X_test) 
    #y_pred_model_KNN_reg = model_KNN_reg.predict(X_test)
    

    model_choisi = st.selectbox(label = "Modèle", options =["Linear Regression", "Decision Tree", "Gradient Boosting regressor", "XGBoost","Grid Search"]) #"Random Forest regressor", "KNN",

    def train_model(model_choisi):
        if model_choisi == "Linear Regression":
            y_pred = y_pred_model_reg_line
        #elif model_choisi == "Random Forest regressor":
        #    y_pred = y_pred_model_Rf_reg
        elif model_choisi == "Decision Tree":
            y_pred = y_pred_model_DecisionTree_regressor
        elif model_choisi == "Gradient Boosting regressor":
            y_pred = y_pred_model_gb_regressor
        elif model_choisi == "XGBoost":
            y_pred = y_pred_model_XGB_reg
        elif model_choisi == "Grid Search":
            y_pred = y_pred_model_Grid_CV
        #elif model_choisi == "KNN":
        #    y_pred = y_pred_model_KNN_reg
        mae = mean_absolute_error (y_test, y_pred)
        #r2 = r2_score(y_test, y_pred)
        st.write(f"La Mean Absolute Error est : {mae:.2f}")

    st.write(train_model(model_choisi))
    if model_choisi == "Decision Tree":
                #st.image(image, caption='Arbre de décision', use_column_width=True)
                image_path = r"..\Images\Arbre_de_decision.png"
                image = Image.open(image_path)
                # Afficher l'image avec Streamlit
                st.image(image, caption='Arbre de décision', use_column_width=True)
    



    


# Deep Learning
elif section == "Deep Learning":
    #.title("Deep Learning")
    #st.write("Application de modèles de deep learning.")
    st.markdown("<h2 style='text-align: center; color: black;'>Modèle de Deep Learning</h2>", unsafe_allow_html=True)

    X_test = np.loadtxt(r"../data/LFB_Test.csv", delimiter=",", dtype=float)
    y_test = np.loadtxt(r"../data/LFB_Test_y.csv", delimiter=",", dtype=float)

    loaded_model = tf.keras.models.load_model(r"../models/model_2024-04-29", compile = False)
    y_pred = loaded_model.predict(X_test)

    st.metric(label="MAE DL Test : ", value=str(round(mean_absolute_error(y_test, y_pred),1))+" secondes")

    #st.metric(label="MSE DL Test : ", value=str(round(mean_squared_error(y_test, y_pred),1)))

    # Ajoutez ici votre code de deep learning

# Perspectives
elif section == "Perspectives":
    st.title("")
    st.markdown("<h1 style='text-align: center; color: blue;'>Conclusion & Perspectives</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; color: black;'>Conclusion</h2>", unsafe_allow_html=True)


    # Paragraphe 1 avec des listes à puces pour les éléments
    paragraphe1 = """
    Au travers de ce projet, nous avons pu mettre en œuvre les principes de data science que sont
    l’analyse et l’exploitation des données afin de pouvoir créer un modèle de prédiction pour aide à la
    décision, de bout en bout sur un sujet dédié très concret (utilisation de données liées aux interventions de la 
    brigade des pompiers de Londres).

    Ainsi, notre projet s'est décomposé en plusieurs grandes étapes, à travers des aspects vus au cours de notre formation :
    - l'exploration des datasets afin de découvrir et s'approprier les données à notre disposition
    - le traitement et nettoyage des données
    - l'analyse approfondie des données afin de consolider nos éléments d’exploitation et en tirer plusieurs enseignements ou orientations d’investigation
    - la réalisation d'une étude de séries temporelles pour vérifier des éléments de saisonnalité
    - la mise en pratique des principes de la réduction de dimensions afin de répondre à des problématiques de temps de chargement et de capacité mémoire
    - l'élaboration de modèles de Machine Learning relatifs à de la régression
    - l’utilisation d’outils de Deep Learning sur des données tabulaires, dans le but d'obtenir des résultats plus performants
    """
    st.markdown(paragraphe1)

    # Sous-titre
    st.markdown("<h2 style='text-align: left; color: black;'>Perspectives suite au projet</h2>", unsafe_allow_html=True)

    # Sous-sous-titre
    st.markdown("<h3 style='text-align: left; color: black;'>Séries temporelles</h3>", unsafe_allow_html=True)

    # Paragraphe 2 avec texte en gras
    paragraphe2 = """
    L'étude temporelle nous a permis de mettre en évidence une progression croissante du nombre d'appels.
    Cette prévision peut avoir son importance sur les besoins en ressources des brigades de Londres (que ce soit pour englober 
    la masse d’appels en augmentation mais également en ressources logistiques et matérielles pour maintenir le niveau 
    d’intervention évalué).

    Il sera notamment intéressant dans la suite de ce projet de pouvoir **faire une évaluation croisée entre les données prédites 
    et les données réelles sur les années à venir**, afin de pouvoir réinterroger notre modèle et l’optimiser.
    """
    st.markdown(paragraphe2)

    # Sous-sous-titre
    st.markdown("<h3 style='text-align: left; color: black;'>Enrichissement de données</h3>", unsafe_allow_html=True)

    # Paragraphe 3 avec texte en gras
    paragraphe3 = """
    Dans une volonté d’optimiser les performances de notre modèle, il serait
    intéressant que d’autres données (dont nous n’avions pas accès) puissent être incluses.
    Nous pensons notamment aux **notions de trafic routier des quartiers de Londres** (existe-t-il un fichier
    pouvant recenser/estimer le nombre de véhicules par secteur en fonction du jour et de l’heure ?
    l’identification des pics de trafic et donc de risque d’engorgement des voies de circulations pouvant
    engendrer des retards potentiels ?).
    """
    st.markdown(paragraphe3)

    # Paragraphe 4 avec texte en gras
    paragraphe4 = """
    Les données d’API de Google Maps sont intéressantes mais trop chères pour la taille du jeu de données.
    Au niveau des bases de données des pompiers, nous pensons peut-être judicieux d’inclure de nouveaux paramètres tels 
    que **la gravité de l’évènement signalé** par exemple.

    Nous présageons que plus le nombre de casernes mises en alerte est important, plus l’évènement est
    critique mais une variable fiabilisée sur cet aspect pourrait être un facteur d’étude ciblé important.
    """
    st.markdown(paragraphe4)

    
