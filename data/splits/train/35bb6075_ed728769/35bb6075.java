import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.*;

public class Main {

    static final long M = 1000000007;

    // use main for only io
    public static void main(String args[]) {
        FastReader io = new FastReader();
        new Solver().solve(io);
    }
}

//class Pair{
//    int key;String value;
//
//    public Pair(int key, String value){
//        this.key = key;
//        this.value = value;
//    }
//}



class Solver {
    static final int M = 998244353;

    void solve(FastReader io) {
       int t = io.nextInt();

       while(t-- > 0){
           int n = io.nextInt();

           String s[] = new String[n];

           for(int i = 0;i < n;i++)s[i] = io.nextLine();

           int ans = 0;

           for(int i = 0;i < 5;i++){
               int count[] = new int[n];

               for(int j = 0;j < n;j++){
                   int freq = 0;
                   for(int k = 0;k < s[j].length();k++){
//                       System.out.println(s[j].charAt(k) - 'a');
                       if((s[j].charAt(k) - 'a') == i){

                           freq++;
                       }
                   }
//                    System.out.println(i + " " + freq);
                   count[j] = 2*freq - s[j].length();
               }


               Arrays.sort(count);
//                for(int it : count)System.out.print(it + " ");
//                System.out.println();

               int curr = 0;
               int j = n - 1;
               for(;j >= 0 && (curr + count[j] > 0);j--){
                   curr += count[j];
               }

               ans = Math.max(ans, n - j - 1);

           }

           System.out.println(ans);
       }
    }

    // returns the first key greater than or equal to val
    

    // returns the first key strictly greater than val
    
    

    

    

    

    

    

    
}



