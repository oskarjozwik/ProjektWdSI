# ProjektWdSI
Model oceniający jadalność grzybów na podstawie ich cech

##1. Opis rzeczywistego problemu
Wybrano problem oceny jadalności grzybów na podstawie jego cech. Na podstawie następujących cech:
-środowisko występowania,
-liczebność,
-kształt i kolor kapelusza,
-zapach,
-wielkość i kolor blaszek,
-kształt trzonu,
-liczba pierścieni.
Model ma oceniać czy grzyb jest trujący czy bezpieczny do spożycia, zwracając na podstawie podanych przez użytkownika danych wejściowych klasę grzyba: jadalny lub trujący. Problem ten jest istotny z uwagi na niemałą popularność zbierania grzybów, taki model może pomóc w uniknięciu pomyłek w ocenie grzyba, a w efekcie zapobiec powstaniu zagrożenia dla zdrowia lub nawet życia osoby go spożywającej.
##2. Znane koncepcje rozwiązania tego problemu 
Jako przykładowe metody rozpoznawania grzybów bezpiecznych do spożycia wybrano trzy algorytmy uczenia nadzorowanego: drzewa decyzyjne, lasy losowe oraz regresja logistyczna. 
Każdy z tych algorytmów jest w stanie osiągnąć zadowalającą dokładność dla wybranej bazy danych.
##3. Wybrana koncepcja
Do rozwiązania tego problemu wybrano algorytm lasów losowych - rf (random forest classifier). Jako dane użyte do wytrenowania modelu wybrano publicznie dostępną bazę: https://www.kaggle.com/datasets/uciml/mushroom-classification/data. Algorytm rf wybrano, ze względu na charakterystykę problemu i zbioru danych dla których model miał wysoką skuteczność klasyfikacji, odporność na przeuczenie, dobrą obsługę danych kategorycznych, możliwość modelowania złożonych, nieliniowych zależności. Algorytm ten tworzy zespoły losowych drzew decyzyjnych na podstawie zbioru treningowego, które po uśrednieniu dają wynik w postaci binarnej oceny jadalności grzyba. W wyniku tego, że pewne cechy niemal jednoznacznie wskazują na klasę grzyba zastosowanie algorytmu rf sprawdza się doskonale. Do realizacji algorytmu zastosowano język Python oraz biblioteki, między innymi: scikit-learn oraz pandas. Przygotowano również interfejs użytkownika, z poziomu którego można wprowadzić cechy ocenianego grzyba i odczytać jak został zaklasyfikowany. Do jego implementacji zastosowano bibliotekę customtkinter. W celu przeprowadzenia testów należy podzielić zbiór na dane treningowe i testowe (wybrano proporcje 80:20), następnie dla wytrenowanego już modelu przeprowadzić predykcję dla danych wejściowych ze zbioru testowego. Uzyskane oceny jadalności należy porównać z faktycznymi z danych wyjściowych zbioru testowego i obliczyć macierz pomyłek oraz dokładność. Podczas realizacji tej koncepcji napotkano nietypową sytuację w postaci 100% dokładności modelu wytrenowanego na danych ze zbioru treningwego dla zbioru danych testowych niezależnie od tego, gdzie trafiły poszczególne rekordy. Pomimo teoretycznie idealnej dokładności model ten jest podatny na dwa istotne problemy. Pierwszym z nich jest mała skuteczność jeżeli podejmie się próbę identyfikacji grzyba, który nie byłby dobrze reprezentowany w bazie użytej do uczenia modelu. Można go rozwiązać upewniając się że baza jest odpowiednio rozbudowana. Drugim jest zawodność potencjalnego użytkownika w poprawnym wprowadzaniu cech ocenianego grzyba. Ten problem jest nierozwiązywalny dla użytej metody i wymagałby znacznego skomplikowania rozwiązania. Przykładowo dodania identyfikacji gatunku i wyświetlanie gatunków często z nim mylonych.
