# Mini_Projet_DeepL

Dans le cadre du cours "Machine learning & differentiable programming" , il nous a été demandé de travailler par binome sur un mini-projet qui a comme objectif d'entrainer un modèle sur une dataset connue et puis d'analyser les résultats.

Pour ceci, voici les étapes qu'on réalisé tout au long de ce mini-projet:

1. Choix de la dataset et étapes de preprocessing
2. Rappel sur le modèle choisi
3. Implémentation des codes
4. Analyse des résultats
5. Conclusion

# I. Choix de la dataset et différents étapes de preprocessing :

Nous avons choisi la dataset "Stock quotidien dans les stockages de gaz" de la source "https://www.data.gouv.fr" . Il s'agit d'une grande base de données qui contient des données sur le débit quotidien des stockages de gaz à partir de novembre 2010 jusqu'à novembre 2022.

Nous avons décidé tout d'abord de passer par une étape primordiale qui est le nettoyage des données.
Ayant comme objectif de prédire le stock de fin de journée en fonction d'une longeur de séquence; c'est-à dire en se basant sur un nombre de jours, on essaie d'estimer le prochain jour.
On s'est intéressé plus particulièrement aux données dont le type de puits est "centre" .

On a commencé par trier notre dataset par order des dates, puis on a filtré les données de tel sorte que la première expérience se fait sur des données entre "2019-12-31" et 2021-1-1 (1 année), et pour la deuxième expérience, c'est entre "2017-12-31" et 2021-1-1 (3 années).

- Pour la première expérience (1 an) : On a décidé de réaliser deux expériences encore:

1. predire le 11ème jour
2. prédire le 31ème jour

- Pour la deuxième expérience (3 ans) : predire le 11ème jour

Ensuite, nous avons dévisé notre jeu de données en 80% comme train dataset et 20% comme test dataset.

Enfin, nous avons fait appel à MinMaxScaler() pour faire scaler les deux datasets.
Cette étape est primordiale parce que dans notre cas, car la dataset choisie contient des valeurs avec des unités différentes.

# II. Rappel sur le modèle choisi :

Face à ce problème de prédiction en fonction d'un nombre de séquence (10 jours ou 30jours), nous avons décidé d'utiliser un modèle qu'on avait bien détaillé en cours qui est le modèle LSTM.
Pour rappel, la particularité des LSTMS c'est qu'ils sont capables de modéliser des dépendances à très long terme. Ceci vient du fait qu'ils possèdent une mémoire interne appelée cellule (ou cell). Cette dernière permet de maintenir un état aussi longtemps que nécessaire. Cette cellule consiste en une valeur numérique que le réseau peut piloter en fonction des situations.

La cellule peut être pilotée par trois portes de contrôle, ou gates :

    1. gate d'entrée décide si l'entrée doit modifier le contenu de la cellule .

    2. gate d'oubli décide s'il faut remettre à 0 le contenu de la cellule .

    3. gate de sortie décide si le contenu de la cellule doit influer sur la sortie du neurone.

Dans notre code, le modèle a été conçu avec 2 couches cachées.

# 3. Implementation du code :

Pour l'implementation du code, vous pouvez voir le fichier "main.py". On peut résumer cette implémentation par trois points :

1.Importation de la dataset
2.Preprocessing de la dataset
3.Division de la dataset en dataset train et dataset test (80/20)
4.Définition de la fonction denetre_glissante() : Celle-ci va donner la main à l'utilisateur de choisir le nombre de séquence qu'il souhaite; ca pourrait etre 1 jours ou 10 jours ... Du coup, ceci va nous permettre de distinguer entre les deux expériences mentionnées auparavant.
4.Création de la classe PredicteurStockQuotidien(nn.Module) qui contient notre modèle LSTM
5.Définition des fonctions train() et test()

# 4. Analyse des résultats :

# A) Première expérience : avec longeur de séquence égale à 10jours

A.a) Les curves :

En lancant le code "main.py" et en particuliant en faisant appel à la fonction using_LSTM() :
On remarque que notre modèle résulte des courbes d'apprentissage de bon ajustement (Good Fit learning curves) et ceci s'explique par le fait que les deux courbes continuent à décroitre jusqu'à un point de stabilisation (ceci se voit mieux avec un nombre epoch de 50, voir image2). On remarque également qu'entre les deux courbes on trouve une marge, appelé souvent gap de généralisation qui fait référence à la capacité de votre modèle à s'adapter correctement à de nouvelles données qui n'étaient pas visibles au préalable.

En ce qui concerne notre modèle, on peut dire qu'il n'est pas trop stable contre ce qu'on appelle "data noise" ; parce que comme on le voit dans les résulats; la courbe de test-loss ne decroit pas d'une facon souple.

![image](Image_1_experience_1.png)

![image](Image_2_experience_1.png)

A.b) Temps d'entrainement :

Avec le LSTM : en réalisant 5 expériences sur notre code, on a remarqué que le temps d'entrainement est entre 10s et 20s et que les courbes on été différentes pour chaque plot .Ce qui est normal car les LSTMS sont stochastiques.

# B) Deuxième expérience :

On garde les memes valeurs pour les paramètres (nombr d'epoch, learning_rate..)
B.a) Les curves :

En lancant le code "main.py" et en particuliant en faisant appel à la fonction using_LSTM() :
On remarque que notre modèle résulte un modèle underfit et ceci s'explique par le fait que la test loss est supérieur presque 20fois plus que la train loss . On remarque également qu'entre les deux courbes on trouve une large marge.
On remarque que le noise persiste toujours dans notre modèle ce qui est normal puisque c'est du a la nature de la dataset.
![image](Image_1_experience_2.png)

B.b) Temps d'entrainement :

Avec le LSTM : en réalisant 5 expériences sur notre code, on a remarqué que le temps d'entrainement est entre 20s et 30s et que les courbes on été différentes pour chaque plot .Ce qui est normal car les LSTMS sont stochastiques.

Face à un modèle underfit, on avait pensé cette fois-ci à augmenter le temps d'entrainement en augmentant le nombre d'epoches. On remarque alors qu'on peut éviter l'underfitting en trouvant un nombre idéal comme nombre d'epoch ; dans notre c'etait pour 100 epoch.
Bien évidement, avec un nombre aussi grand le temps d'entrainement était plus supérieur.

![image](Image_2_experience_2.png)

#### Conclusion :

Ce mini-projet nous a vraiment à aider à appliquer les notions du cours que nous avons vu. Aujourdh'ui on est plus à l'aise voire capable de passer par les différentes étapes pour analyser et prendre le recul sur les modèles du deep learning qu'on choisi. C'est une occasion également pour revoir plusieurs notions et comprendre les différentes courbes, paramètres et hyperparamètre utilisés dans ce cadre.
Sans oublier que grace a ce cours, on a pu améliorer notre manière d'écrire et déposer les codes sur github.

# Bibliographie :

Pour la réalisation de ce travail, on s'est appuyé sur les ressources suivantes :

https://members.loria.fr/CCerisara/#courses/machine_learning/
https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
