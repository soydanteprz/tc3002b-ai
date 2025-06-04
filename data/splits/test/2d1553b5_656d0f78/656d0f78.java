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
    
   
   
    public static void main(String[] args)
    {
        FastScanner input = new FastScanner();
        int tc = input.nextInt();
        work:
        while (tc-- > 0) {
             int n = input.nextInt();
       int ans[] = new int[n];
        for (int i = 1; i <=n; i++) {
            if(ans[i-1]!=0)
            {
//                System.out.println("NO");
                continue ;
            }
            HashSet<Integer> set = new HashSet<>();
            ArrayList<Integer> a = new ArrayList<>();
            while(true)
            {
                System.out.println("? "+i);
                int value = input.nextInt();
                if(set.contains(value))
                {
                    break;
                }
                a.add(value-1);
                set.add(value);
                
            }
            for (int j = 0; j <a.size(); j++) {
                ans[a.get(j)] = a.get((j+1)%a.size())+1;
            }
//            System.out.println(a);
        }
        System.out.print("! ");
        for (int an : ans) {
            System.out.print(an+" ");
        }
        System.out.println("");   
        }
    }

    

}
