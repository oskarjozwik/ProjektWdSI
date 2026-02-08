# ProjektWdSI
Model oceniający jadalność grzybów na podstawie ich cech

## 1. Opis rzeczywistego problemu

Wybrano problem oceny jadalności grzybów na podstawie jego cech. Na podstawie następujących cech:

-środowisko występowania,

-liczebność,

-kształt i kolor kapelusza,

-zapach,

-wielkość i kolor blaszek,

-kształt trzonu,

-liczba pierścieni.

Model ma oceniać czy grzyb jest trujący czy bezpieczny do spożycia, zwracając na podstawie podanych przez użytkownika danych wejściowych klasę grzyba: jadalny lub trujący. Problem ten jest istotny z uwagi na niemałą popularność zbierania grzybów oraz spore ryzyko związane z ich spożywaniem. Taki model może pomóc w uniknięciu pomyłek w ocenie jadalności grzyba, a w efekcie zapobiec powstaniu zagrożenia dla zdrowia lub nawet życia osoby go spożywającej.

## 2. Znane koncepcje rozwiązania tego problemu 

Do znanych metod rozwiązywania tego problemu, możliwych do napotkania w literaturze (https://ieeexplore.ieee.org/abstract/document/9797212) należą algorytmy: naiwny klasyfikator bayesowski - nb (z ang. Naive Bayes), algorytm drzew decyzyjnych - dt (z ang. Decission Tree), algorytm lasów losowych - rf (z ang. random forest), algorytm wektorów nośnych - svm (z ang. Support Vector Machine) oraz algorytm AdaBoost - ab (z ang. Adaptive Boosting). 

Alogrytm nb jest klasyfikatorem opartym na twierdzeniu Bayesa należący do klasyfikatorów probabilistycznych. Klasyfikator oblicza wartości prawdopodobieństwa dla każdej klasy i dąży do znalezienia najbardziej prawdopodobnej klasy dla każdej klasyfikowanej danej. Jest on algorytmem stosunkowo prostym i o małej złożoności obliczeniowej, z czego wynika popularność jego implementacji.

Algorytm dt jest metodą nadzorowanego uczenia maszynowego, stosowana do klasyfikacji i regresji. Drzewa decyzyjne uczą się prostych reguł decyzyjnych wyprowadzonych z zastosowanej do uczenia bazy danych w celu stworzenia modelu przewidującego wartość zmiennej docelowej. Drzewo decyzyjne uczy się aproksymować rozwiązanie problemu na podstawie danych za pomocą szeregu reguł typu if–then–else. Wraz ze wzrostem głębokości drzewa reguły decyzyjne stają się bardziej złożone i w efekcie skuteczność modelu ulega poprawie.

Algorytm svm jest algorytmem klasyfikacyjnym opartym na statystycznej teorii uczenia się. Działanie SVM polega na estymacji najlepszej funkcji decyzyjnej, która potrafi rozdzielić dwie klasy, wyznaczając jak największy margines, czyli odległość danych od hiperpłaszczyzny rozdzielającej je. Wektory nośne są kluczowymi dla tego rozdziału punktami określającymi jak przebiega podział i to one są dla tego algorytmu najważniejsze.
 
Algorytm ab ma tworzy silny model poprzez łączenie wielu słabszych algorytmów uczących, z których każdy kolejny jest uczony z uwzględnieniem wyników poprzedniego i ze szczególnym naciskiem na przykłady, z których identyfikacją miał on problem. Pojedyńcze modele zazwyczaj oparte są na prostych drzewach decyzyjnych a ostateczna klasyfikacja jest wynikiem uśrednienia wyników ze wszystkich modeli (wykonania głosowania).

## 3. Wybrana koncepcja

Do rozwiązania tego problemu wybrano algorytm lasów losowych - rf (random forest classifier).  Jako dane użyte do wytrenowania modelu wybrano publicznie dostępną bazę: https://www.kaggle.com/datasets/uciml/mushroom-classification/data. Algorytm rf wybrano, ze względu na charakterystykę problemu i zbioru danych dla których model miał wysoką skuteczność klasyfikacji, odporność na przeuczenie, dobrą obsługę danych kategorycznych, możliwość modelowania złożonych, nieliniowych zależności. Algorytm ten tworzy zespoły losowych drzew decyzyjnych na podstawie zbioru treningowego, które po uśrednieniu ich ocen dają ostateczny wynik w postaci binarnej oceny jadalności grzyba, jest więc on algorytmem pokrewnym do algorytmu dt opisanego w znanych z literatury rozwiązaniach problemu. W wyniku tego, że pewne cechy niemal jednoznacznie wskazują na klasę grzyba zastosowanie algorytmu rf sprawdza się doskonale. Do realizacji algorytmu zastosowano język Python oraz biblioteki, między innymi: scikit-learn oraz pandas. Przygotowano również interfejs użytkownika, z poziomu którego można wprowadzić cechy ocenianego grzyba i odczytać jak został zaklasyfikowany. Do jego implementacji zastosowano bibliotekę customtkinter. W celu przeprowadzenia testów należy podzielić zbiór na dane treningowe i testowe (wybrano proporcje 80:20), następnie dla wytrenowanego już modelu przeprowadzić predykcję dla danych wejściowych ze zbioru testowego. Uzyskane oceny jadalności należy porównać z faktycznymi z danych wyjściowych zbioru testowego i obliczyć macierz pomyłek oraz dokładność. Podczas realizacji tej koncepcji napotkano nietypową sytuację w postaci 100% dokładności modelu wytrenowanego na danych ze zbioru treningwego dla zbioru danych testowych niezależnie od tego, gdzie trafiły poszczególne rekordy. Pomimo teoretycznie idealnej dokładności model ten jest podatny na dwa istotne problemy. Pierwszym z nich jest mała skuteczność jeżeli podejmie się próbę identyfikacji grzyba, który nie byłby dobrze reprezentowany w bazie użytej do uczenia modelu. Można go rozwiązać upewniając się że baza jest odpowiednio rozbudowana. Drugim jest zawodność potencjalnego użytkownika w poprawnym wprowadzaniu cech ocenianego grzyba. Ten problem jest nierozwiązywalny dla użytej metody i wymagałby znacznego skomplikowania rozwiązania. Przykładowo dodania dodatkowo identyfikacji gatunku i wyświetlanie gatunków często z nim mylonych, co wymagałoby znaczącego rozbudowania używanej bazy danych oraz zwiększenia wkładu użytkownika.
