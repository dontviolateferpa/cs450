# Results

## 01 - The 5 rules you can find with the highest support

### Support = 0.00233858668022369

```text
    lhs                                                         rhs                support     confidence lift     count
    {tropical fruit,herbs}                                   => {whole milk}       0.002338587 0.8214286  3.214783 23
    {herbs,rolls/buns}                                       => {whole milk}       0.002440264 0.8000000  3.130919 24
    {hamburger meat,curd}                                    => {whole milk}       0.002541942 0.8064516  3.156169 25
    {other vegetables,curd,domestic eggs}                    => {whole milk}       0.002846975 0.8235294  3.223005 28
    {citrus fruit,tropical fruit,root vegetables,whole milk} => {other vegetables} 0.003152008 0.8857143  4.577509 31
    {citrus fruit,root vegetables,other vegetables,yogurt}   => {whole milk}       0.002338587 0.8214286  3.214783 23
```

## 02 - The 5 rules you can find with the highest confidence

```text
    lhs                                                                 rhs                support     confidence lift        count
    {root vegetables,whipped/sour cream,flour}                       => {whole milk}       0.001728521 1          3.913649025 17
    {root vegetables,other vegetables,yogurt,oil}                    => {whole milk}       0.001423488 1          3.913649025 14
    {rice,sugar}                                                     => {whole milk}       0.001220132 1          3.913649025 12
    {other vegetables,butter,whipped/sour cream,domestic eggs}       => {whole milk}       0.001220132 1          3.913649025 12
    {citrus fruit,tropical fruit,root vegetables,whipped/sour cream} => {other vegetables} 0.001220132 1          5.168155544 12
```

## 03 - The 5 rules you can find with the highest lift

```text
   lhs                                                           rhs                support     confidence  lift        count
   {liquor,red/blush wine}                                    => {bottled beer}     0.001931876 0.9047619   11.235269   19
   {citrus fruit,other vegetables,soda,fruit/vegetable juice} => {root vegetables}  0.001016777 0.909090909 8.340400271 10
   {tropical fruit,other vegetables,whole milk,yogurt,oil}    => {root vegetables}  0.001016777 0.909090909 8.340400271 10
   {citrus fruit,grapes,fruit/vegetable juice}                => {tropical fruit}   0.001118454 0.846153846 8.063878951 11
   {other vegetables,whole milk,yogurt,rice}                  => {root vegetables}  0.00132181  0.866666667 7.951181592 13
```

## 04 - The 5 rules you think are the most interesting

```text
   lhs                                            rhs                support     confidence  lift        count
   {rice,bottled water}                        => {whole milk}       0.001220132 0.923076923 3.6125991   12
   {tropical fruit,dessert,whipped/sour cream} => {other vegetables} 0.001118454 0.916666667 4.737475915 11
   {sausage,pip fruit,sliced cheese}           => {yogurt}           0.001220132 0.857142857 6.144314869 12
   {tropical fruit,whole milk,butter,curd}     => {yogurt}           0.001220132 0.857142857 6.144314869 12
   {rolls/buns}                                => {canned beer}      0.01128622  0.06135987  0.7898878   111
```
