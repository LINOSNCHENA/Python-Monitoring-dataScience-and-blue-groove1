def products = [
    1: [Product:'A', Group: 'G1', Cost:20.1],
    2: [Product:'B', Group: 'G2', Cost:98.4],
    3: [Product:'C', Group: 'G1', Cost:49.7],
    4: [Product:'D', Group: 'G3', Cost:35.8],
    5: [Product:'E', Group: 'G3', Cost:105.5],
    6: [Product:'F', Group: 'G1', Cost:55.2],
    7: [Product:'G', Group: 'G1', Cost:12.7],
    8: [Product:'H', Group: 'G3', Cost:88.6],
    9: [Product:'I', Group: 'G1', Cost:5.2],
    10: [Product:'J', Group: 'G2', Cost:72.4],]
    

for(int i=1;i<11;i++){
if(products[i].Cost>=00&&products[i].Cost<25) { products[i].put('Price',products[i].Cost*1.2) }
if(products[i].Cost>=25&&products[i].Cost<50) { products[i].put('Price',products[i].Cost*1.3) }
if(products[i].Cost>=50&&products[i].Cost<75) { products[i].put('Price',products[i].Cost*1.4) }
if(products[i].Cost>=75&&products[i].Cost<100) { products[i].put('Price',products[i].Cost*1.5) }
if(products[i].Cost>=100&&products[i].Cost<999) { products[i].put('Price',products[i].Cost*1.6) }
}


def CX1 = products.findAll { it.value.Group == 'G1' }
def CX2 = products.findAll { it.value.Group == 'G2' }
def CX3 = products.findAll { it.value.Group == 'G3' }

def CY1= CX1.collect{it.value.Price}
def CY2= CX2.collect{it.value.Price}
def CY3= CX3.collect{it.value.Price}

 def G1 = CY1.sum()/CY1.size()
 def G2 = CY2.sum()/CY2.size()
 def G3 = CY3.sum()/CY3.size()
 def result=[37.498,124.48,116.08];

products.each { key, value ->
    println "ProductX: $key Price: $value"
}

println '************************************'
println CY1.sum()/CY1.size()
println CY2.sum()/CY2.size()
println CY3.sum()/CY3.size()
println "***********************************"

assert result==[  
          G1,G2,G3
          ] : "It does'nt work"
println "It works"



