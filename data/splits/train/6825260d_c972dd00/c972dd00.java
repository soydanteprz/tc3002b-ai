import java.awt.Container;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.math.BigInteger;

import java.util.*;

public class Main
{
    public static int check(String a,String b)
    {
        int zero1 = 0,one0 = 0;
        for (int i = 0; i <a.length(); i++) {
            if(a.charAt(i)=='1'&&a.charAt(i)!=b.charAt(i))
            {
                one0++;
            }
            else if(a.charAt(i)=='0'&&a.charAt(i)!=b.charAt(i))
            {
                zero1++;
            }
        }
        if(zero1!=one0)
        {
            return -1;
        }
        else
            return zero1+one0;
    }
    public static int changeandAns(String a ,String b,int index)
    {
        
      char newa[] = new char[a.length()];
        for (int i = 0; i <a.length(); i++) {
            if(i!=index)
            {
                if(a.charAt(i)=='0')
                {
                    newa[i]='1';
                    
                }
                else
                {
                    newa[i]= '0';
                }
            }
            else
            {
                newa[i]= a.charAt(i);
            }
        }
        String ra = new String(newa);
        return check(ra, b);
        
    }
    
    public static void main(String[] args)
    {
        FastScanner input = new FastScanner();
       
        int tc = input.nextInt();
        StringBuilder result = new StringBuilder();
        work:
        while (tc-- > 0) {
            int n = input.nextInt();
        String a = input.next();
        String b = input.next();
        int ans = Integer.MAX_VALUE;
        //no changes
        int noChange = check(a,b);
        if(noChange!=-1)
        {
            ans = Math.min(ans, noChange);
        }
        //1 - 1
        boolean have = false;
        int index = -1;
            for (int i = 0; i <n; i++) {
            if(a.charAt(i)=='1'&&b.charAt(i)==a.charAt(i))
            {
                have  = true;
                index = i;
                break;
                
            }
        }
            if(have)
            {
                
                
                int onetone = changeandAns(a,b,index);
                if(onetone!=-1)
                {
                    ans = Math.min(ans,1+ onetone);
                }
                
                
            }
            //1-0
            have = false;
            index = -1;
            
                    for (int i = 0; i <n; i++) {
            if(a.charAt(i)=='1'&&b.charAt(i)!=a.charAt(i))
            {
                have  = true;
                index = i;
                break;
                
            }
        }
            if(have)
            {
                
                
                int onetozero = changeandAns(a,b,index);
                if(onetozero!=-1)
                {
                    ans = Math.min(ans, 1+onetozero);
                }
                
                
            }
            if(ans==Integer.MAX_VALUE)
            {
               result.append("-1\n");
            }
            else
            {
                result.append(ans+"\n");
            }
        }
        System.out.println(result);
        

    }

    

}
