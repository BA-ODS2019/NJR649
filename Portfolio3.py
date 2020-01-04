import requests
import pandas as pd
from numpy.core.defchararray import capitalize
from pandas.io.json import json_normalize

# Efter vi har importeret de påkrævede elementer opretter vi en variable som indeholder vores søgnings-url.
smk_search_url = 'https://api.smk.dk/api/v1/art/search/?keys=kat&qfields=titles&offset=0&rows=500'

# Derefter opstiller vi de relevante parametre
params = {
    'q': 'kat',
    'items': 'items',
    'encoding': 'json',
    'n': 500
}

# Her bruges den tidligere importerede 'requests' til at hente vores søgning ned gennem SMK's API.
response = requests.get(smk_search_url, params=params)

# Herefter laves svaret til json og pandas 'json_normalize' bruges til at 'beautify' vores resultat og få det ind i en
# pandas-dataframe. Derefter printer vi vores nye dataframe.
json = response.json()
df = json_normalize(json['items'])

# Her bruger vi dtypes-funktionen til at finde ud af hvilke forskellige datatyper sættet indeholder.
print(df.dtypes)

# Her bruger vi shape-funktionen for at finde ud af hvor mange kolonner og rækker vores dataframe har.
print(df.shape)

# Her fjerner vi de kolonner som vi, efter at undersøge dataframen, har vurderet ikke er væsentlige for for datasæt.
removed_columns = [
    'created',
    'modified',
    'frame_notes',
    'object_url',
    'iiif_manifest',
    'work_status',
    'image_mime_type',
    'image_iiif_id',
    'image_iiif_info',
    'production_dates_notes',
    'object_history_note',
    'content_description',
    'distinguishing_features',
    'exhibitions',
    'current_location_name',
    'content_subject',
    'inscriptions'
]
df.drop(removed_columns, inplace=True, axis=1)

# Først laver vi 2 variabler som indeholder henholdsvis antallet af gange de teknikkerne brugt er (teknikantal)
# og teknikkens relative anvendelelse ift. hele datasættet i procent (teknikprocent).
teknikantal = df['techniques'].value_counts()
teknikprocent = df['techniques'].value_counts(normalize=True, dropna=True).mul(100).round(1).astype(str) + '%'

# Her oprettes variablen 'værdier' som tager alle navnene på de forskellige teknikker og tilføjer dem til en liste.
værdier = df['techniques'].value_counts().keys().tolist()

# Til sidst opretter vi en ny dataframe baseret på de værdier vi har samlet i de 2 første variabler, samt navnene på
# de forskellige teknikker vi samlede i vores 'værdi' variabel.
teknik = pd.DataFrame(columns=['Antal', 'Procent'], index=pd.Series(værdier), data=zip(teknikantal,teknikprocent))

# Nu er der bare tilbage at printe vores nye datasæt og analysere det.
print(teknik)

# For at lave et håndterbart cirkeldiagram, fjerner jeg de teknikker som udgør mindre end 1%.
# Dette gøres ved først at lave en dataframe lignende den tidligere variable 'teknikprocent', bare kun indeholdende
# procenterne i form af floats, altså uden transformeringen til en string og tilføjelse af '%'-tegnet.
teknikprocent_floats = df['techniques'].value_counts(normalize=True, dropna=False).mul(100).round(1).tolist()

# Derefter fjernes de teknikker som udgør mindre end 1% af totalen via et for-loop indeholdende et if-loop.
teknikprocent_floats[:] = [x for x in teknikprocent_floats if x >= 1]

# Herefter kan vi printe værdierne for at sikre os at de ser ud som det skal.
print(teknikprocent_floats)

# Nu kan vi tage vores liste med værdier og hun hente det antal som der er brug for, altså som er lig længden af vores
# liste indeholdende teknikprocent floats.
værdier_mindst_en = værdier[:len(teknikprocent_floats)]

# For at samle resten af værdierne, altså dem under 1.0 som vi smed ud, kan vi foretage dette regnestykke.
teknikprocent_under_en_samlet = abs(sum(teknikprocent_floats) - 100)

# Derefter kan vi tilføje summen af de resterende værdier tilbage på listen
teknikprocent_floats.append(teknikprocent_under_en_samlet)

# Nu tilføjer vi en string som repræsentere den værdi vi tilføjede før, altså summon af de 'små' værdier.
værdier_mindst_en.append('Andre')

# Nu kan vi lave vores dataframe, som skal bruges til tærtediagrammet.
df_teknik = pd.DataFrame({'Teknikker': [31.5, 13.7, 6.1, 2.9, 2.6, 2.6, 2.2, 2.2, 1.9, 1.3, 1.3, 1.0, 1.0, 1.0, 1.0,
                                        teknikprocent_under_en_samlet],
                          'radius': [teknikprocent]}, index=pd.Series(værdier_mindst_en))
# Til slut kan vi tegne vores plot ved hjælp af dataframe.plot.pie funktionen.
teknik_tærtedia = df_teknik.plot.pie(y='Teknikker', figsize=(10, 10))

# Nu kan er diagrammet lavet og kan printes
print(teknik_tærtedia)

# Lad os prøve at gøre det samme med formatet af værkerne i vores dataset.
formatantal = df['image_orientation'].value_counts()
formatprocent = df['image_orientation'].value_counts(normalize=True, dropna=True).mul(100).round(1).astype(str) + '%'
formatnavne = df['image_orientation'].value_counts().keys().tolist()
capitalize(formatnavne)
format = pd.DataFrame(columns=['Antal', 'Procent'], index=capitalize(formatnavne), data=zip(formatantal, formatprocent))
print(format)

# Her viser vi først summene af de 2 typer i et bardiagram og så den fordeleingen i procent i et tærtediagram.
format_bardia = format.plot(kind='bar')
print(format_bardia)
df_format = pd.DataFrame({'Formattyper': formatantal, 'radius': [formatprocent]}, index=formatnavne)
format_tærtedia = df_format.plot.pie(y='Formattyper', figsize=(10, 10))
print(format_tærtedia)


