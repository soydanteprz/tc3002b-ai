import java.io.*;
import java.lang.*;
import java.util.*;
public class c
{

 

public static int[] ia(int n)
{
    int ar[]=new int[n];
    return ar;
}


static class pair implements Comparable<pair>{
    int ht;
    int id;
    pair(int ht, int id)
{
    this.ht=ht;
    this.id=id;
}

}
public static void main(String args[]) throws java.lang.Exception
{
FastScanner sc=new FastScanner();
 int t=sc.ni();
while(t-->0)
{
    int n=sc.ni();
    int m=sc.ni();
    int x=sc.ni();
    int ar[]=ia(n);
for(int i=0;i<n;i++)
{
ar[i]=sc.ni();
}
System.out.println("YES");
PriorityQueue<pair> pq=new PriorityQueue<>();
 for(int i=0;i<m;i++)
 {
pq.add(new pair(0,i+1));
 }                     
 int i=0;
 while(i<n)
 {
     pair pp=pq.remove();
pp.ht+=ar[i];
System.out.print(pp.id+" ");
pq.add(pp);
i++;
 }   
System.out.println();
}
}
}
