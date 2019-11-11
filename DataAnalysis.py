import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv', sep=",")

print("\nHere are some facts about 887 of the passengers that were aboard the RMS Titanic.")
print("\nThe median age of the passengers aboard was " + str(np.median(data["Age"])))
print("\nThe average age of the passengers on board was " + str(np.mean(data["Age"])) + "\n")

# I overstående print funktioner har jeg brugt numpy til at tage henholdsvis medianen og gennemsnittet af kolonnen "Age".

full_names = data["Name"].tolist()
lastnames = []

# Her har jeg først angivet 2 variables, én som har taget kolonnen "Name" fra vores dataframe, og tilføjet dem til en
# liste (full_names). Den næste er en tom liste ved navn "lastnames" som senere bruges til at opbevare alle passageres
# efternavne.

for name in full_names:
    *notlast, last = name.split()
    lastnames.append(last)

# Her bruger jeg et for-loop til at loop'e hvert navn i full_names variable'en igennem og smide alt væk, bortset fra
# det sidste ord i hver string (hvilket i dette tilfælde altid er passagerens efternavn), og tilføjer dem
# til en vores variable "lastnames".

lastnames_df = pd.DataFrame(lastnames)

# Nu laver vi en ny variable "lastnames_df" som gøres til en dataframe med kun vores lastnames-variable.

print("Analysis of passengers' last names:")

print(lastnames_df.describe())

# Til sidst bruger vi pandas describe() funktion til at give os en kort analyse af de indsamlede efternavne, såsom
# at der kun var 664 unikke efternavne ud af de 887 passagerer, eller at det mest almindelige navn var "Anderson", som
# kunne findes hos 9 passagerer.

print("\nIn this dataframe we've included only the last names of the 887 passengers on board."
      "\nBy using Pandas .describe() function, we can see that of the 887 passengers we found 664 unique last names."
      "\nOut of those, the most common last name was Anderson, which was shared by 9 different passengers.")

print("\nThe following table shows the total number of people of each passenger class, as well as how many survived:\n")
pclass_and_survived_data = (pd.pivot_table(data, "Survived", ['Pclass'], aggfunc=[np.sum, len], margins=True))
pclass_and_survived_data.columns = ["Survivors", "Total"]
pclass_and_survived_data['Survival Rate in %'] = pclass_and_survived_data.apply \
    (lambda row: (row.Survivors / row.Total) * 100, axis=1)
print(pclass_and_survived_data)

# Her bruger vi kolonnerne "Survived" og "Pclass" til at oprette en pivot-tabel som viser os bl.a. antallet af
# passagerer på hver rejseklasse, og hvor mange der overlevede (både i alt og også udfra de 3 rejseklasser).
# Dette opnås ved at bruge aggfunc-funktionen til først at udregne summen af overlevende passagerer (altså var markeret
# med et 1-tal i vores dataframe) og derefter inddele dem i kolonner udfra rejseklasse. Vores "Total"-kolonne fås
# ved brug af en anden aggfunc-funktion, nemlig "len", som giver os det totale antal af passagerer indenfor rejseklasse
# 1, 2 og 3. Den nederste "All"-række fås ved at ændre boolean værdien "margins" til "True". Sidst har jeg tilføjet
# "Survival Rate in %"-kolonnen som bruger apply()- og lambda-funktionen til at udregne procentdelen af vores
# "Survivors"-værdi ift. "Total"-værdien. Til sidst printes hele pivot-tabellen ud.
