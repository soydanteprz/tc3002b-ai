import java.io.*;
import java.util.*;
public class Main {
    public static void main(String args[])
    {
        FastReader input=new FastReader();
        PrintWriter out=new PrintWriter(System.out);
        int T=input.nextInt();
        while(T-->0)
        {
            int n=input.nextInt();
            int a[]=new int[n];
            for(int i=0;i<n;i++) a[i]=input.nextInt();
            long i=1,j=(long)1e15;
            while(i<j)
            {
                long mid=(i+j)/2;
                if(fun(a,mid,n))
                {
                    i=mid+1;
                }
                else
                {
                    j=mid;
                }
            }
            out.println(i-1);
        }
        out.close();
    }
    public static boolean fun(int a[],long x,int n)
    {
        long arr[]=new long[n];
        for(int i=0;i<n;i++) arr[i]=a[i];
        long b[]=new long[n];
        for(int i=n-1;i>=2;i--)
        {
            if(b[i]+arr[i]<x) return false;
            long z=Math.max(0,x-b[i]);
            long u=arr[i]-z;
            long y=u/3;
            b[i-1]+=y;
            b[i-2]+=2*y;
        }
        if(arr[0]+b[0]<x || arr[1]+b[1]<x) return false;
        return true;
    }
    
}
