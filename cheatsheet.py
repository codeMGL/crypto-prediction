# Python cheatsheet
"""
TIPOS:
- type(var)
- isinstance(var, tipo)
- id(): Posición de memoria de una variable

RANDOM:
- random(): [0, 1) FLOAT
- randint(start, end): [start, end] INT
- randrange(start?, end, step?): [start, end) INT
- uniform(start, end): [start, end) FLOAT

BUCLES:
- range(start?, end, step?) [start, end)
- for-each: for elt in mi_lista

CADENAS DE CARACTERES:
- str.upper()/str.lower()
- len(str)
- str[inicio:fin]
- ¿ "ej" in str ?

LISTAS:
- mi_lista = [elt0, elt1, elt2]
- mi_lista += [elt3]
- nueva = mi_lista[start:end]
- ¿ elt5 in mi_lista?
- len(mi_lista)
- del(mi_lista[0]) / del(mi_lista[pos1:pos2])
- mi_lista.remove(elt): Error si no está en la lista
- mi_lista.pop(): Elimina y devuelve el último elemento
- mi_lista.pop(pos): Igual que del(mi_lista[pos]), pero lo devuelve
- mi_lista.append(elt4)
- mi_lista.insert(pos, elt): if pos no existe (pos = -1) --> = .append(elt)
- mi_lista.extend(mi_lista2)
- mi_lista.index(elt): Devuelve la posición. Se debe comprobar primero que está en la lista
- mi_lista.index(elt, start, end?)
- mi_lista.count(elt)
- mi_lista.clear()
- mi_lista.reverse()
- mi_lista.sort(): De menor a mayor
- nueva = mi_lista.copy() / Bucle for / mi_lista[:]  / [].extend(mi_lista)

TUPLAS:
- Inmutables, heterogéneas
- mi_tupla = (elt0, elt1, elt2)
- Mismos métodos que las listas

EMPAQUETAR Y DESEMPAQUETAR:
- Empaquetar: Guardar varios valores en una lista/tupla
- Desempaquetar: Guardar los valores de una lista/tupla en variables

CONVERSIONES:
- lista = list(colección)
- tupla = tuple(colección)
- Colecciones: Listas, tuplas, diccionarios, strings

COPIA PROFUNDA DE LISTAS ANIDADAS:
a) Bucles anidados
b) Librería copy: copy.deepcopy()

DICCIONARIOS:
- mi_dic = {"clave1": valor1, "clave2": valor2}
- Nuevas claves: mi_dic["clave3"] = valor3
- Borrar clave: del(mi_dic["clave1"])
- mi_dic.pop(clave): Devuelve el valor, y borra el valor y clave 
- Buscar claves: in / not in
- len(mi_dic)
- dic1.update(dic2): Añade una copia superficial del dic2. Si hay claves comunes, toman el valor de dic2
- nuevo = mi_dic.copy()
- mi_dic.clear()
- elt = mi_dic.get("clave2") --> Es mejor usar mi_dic["clave2"]
- mi_dic.keys(): Devuelve una pseudo-lista con todas las claves
- mi_dic.values(): Devuelve una pseudo-lista con todos los valores
- mi_dic.items(): Devuelve una pseudo-lista con tuplas (clave, valor)

"""
