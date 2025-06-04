import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.TreeMap;
import java.util.PriorityQueue;
import java.util.Collections;
import java.util.Stack;

import java.math.BigInteger;
import java.util.LinkedList;
import java.util.Iterator;

public class First {

    public static void main(String[] args) {
        FastScanner fs = new FastScanner();
        int T = fs.nextInt();
        for (int tt = 0; tt < T; tt++) {

            solve(fs);
        }
        
    }
    static void solve(FastScanner fs)
    {
        int n=fs.nextInt();
        long[] times=takeLong(n, fs);
        long[] damage=takeLong(n, fs);
        long reqTime=times[n-1]-damage[n-1]+1;
        long ans=0;
        for(int i=n-1;i>=0;)
        {
            reqTime=times[i]-damage[i]+1;
            long time=times[i];
            i--;    
            while(i>=0 && times[i]>=reqTime)
            {
                long thisReqTime=times[i]-damage[i]+1;
                reqTime=Math.min(reqTime, thisReqTime);
                i--;
            }
            long x=time-reqTime+1;
            // pn(x);
            ans+=(x*(x+1)/2);



        }
        pn(ans);
    }
    static long MOD=(long)(1e9+7);
    
    
    static void pn(Object o) { System.out.println(o); } 
    
    

    
    
    
    
    static long[] takeLong(int n, FastScanner fs)
    {
        long[] arr=new long[n];
        for(int i=0;i<n;++i)
            arr[i]=fs.nextLong();
        return arr;
    } 
     
     
     
    
    
    
    
    


}

