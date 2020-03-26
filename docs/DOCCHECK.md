Documentation Checking Process(Only for the developers)
==========================================================  

# Why  

It is necessary for all the developers to generate the rst files which can help us check the documents.  

# When  

1. You add a new function to one of the scripts in the {ULTRA/ultra} or its subdirs  
1. You add a new script to {ULTRA/ultra} or its subdirs  
1. You add a new directory to {ULTRA/ultra} or its subdirs  

# How  
## Make sure you have installed sphinx

1. Enter the docs directory  

```
cd {ULTRA/docs}
```  

2. Generate the rst files  

```
sphinx-apidoc -f -o source/ ../ultra
```  

3. Commit
