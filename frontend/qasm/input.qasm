// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
gate majority a,b,c 
{ 
  cx c,b; 
  cx c,a; 
  ccx a,b,c; 
}

gate unmaj a,b,c 
{ 
  ccx a,b,c; 
  cx c,a; 
  cx a,b; 
}

qreg a[2];
qreg b[2];
creg ans[2];

x a[0];    // a = 0001
x b[0];    // b = 1111

majority a[1],b[0],a[0];
cx a[1],a[0];
unmaj a[1],b[1],a[0];

measure b[0] -> ans[0];
