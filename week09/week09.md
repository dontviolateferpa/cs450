# Results

## 01 - The 5 rules you can find with the highest support

### Support = 0.00233858668022369

```text
    lhs                                                         rhs                support     confidence lift     count
[1] {tropical fruit,herbs}                                   => {whole milk}       0.002338587 0.8214286  3.214783 23   
[2] {herbs,rolls/buns}                                       => {whole milk}       0.002440264 0.8000000  3.130919 24   
[3] {hamburger meat,curd}                                    => {whole milk}       0.002541942 0.8064516  3.156169 25   
[4] {other vegetables,curd,domestic eggs}                    => {whole milk}       0.002846975 0.8235294  3.223005 28   
[5] {citrus fruit,tropical fruit,root vegetables,whole milk} => {other vegetables} 0.003152008 0.8857143  4.577509 31   
[6] {citrus fruit,root vegetables,other vegetables,yogurt}   => {whole milk}       0.002338587 0.8214286  3.214783 23 
```

## 02 - The 5 rules you can find with the highest confidence

### Confidence = 0.1395

```text
    lhs    rhs                support   confidence lift count
[1] {}  => {soda}             0.1743772 0.1743772  1    1715 
[2] {}  => {yogurt}           0.1395018 0.1395018  1    1372 
[3] {}  => {rolls/buns}       0.1839349 0.1839349  1    1809 
[4] {}  => {other vegetables} 0.1934926 0.1934926  1    1903 
[5] {}  => {whole milk}       0.2555160 0.2555160  1    2513 
```

## 03 - The 5 rules you can find with the highest lift

### Confidence = 0.001

```text
    lhs                                                           rhs                support     confidence  lift        count
    {liquor,red/blush wine}                                    => {bottled beer}     0.001931876 0.9047619   11.235269   19
    {citrus fruit,other vegetables,soda,fruit/vegetable juice} => {root vegetables}  0.001016777 0.909090909 8.340400271 10
    {tropical fruit,other vegetables,whole milk,yogurt,oil}    => {root vegetables}  0.001016777 0.909090909 8.340400271 10
    {citrus fruit,grapes,fruit/vegetable juice}                => {tropical fruit}   0.001118454 0.846153846 8.063878951 11
    {other vegetables,whole milk,yogurt,rice}                  => {root vegetables}  0.00132181  0.866666667 7.951181592 13
```
