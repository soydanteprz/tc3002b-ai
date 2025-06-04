import java.io.*;
import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collections;
import java.io.InputStreamReader;
import static java.lang.Math.*;
import static java.lang.System.*;

public class Main1 {

     static ArrayList<Integer> list1 = new ArrayList<>() ;
     static ArrayList<Integer> list2 = new ArrayList<>() ;
     static int n , m ;
     static long dp[][] ;

     static long solver(int i , int j ){
         // i = empty chairs

         if (j == m)return 0 ;
         int tt1 = n-i ;
         int tt2 = m-j ;
         if (n-i < m-j)return  Long.MAX_VALUE/2  ;
         if ( dp[i][j] != -1 )return dp[i][j] ;

        long a = solver(i+1 , j) ;
        long b = abs( list1.get(i) - list2.get(j)) + solver(i+1 , j+1) ;

        return dp[i][j] = min(a , b) ;

     }



    public static void main(String[] args) throws IOException {

//        try {


            FastScanner in = new FastScanner();
            PrintWriter out = new PrintWriter(System.out);


            int N = in.nextInt() ;
            int a[] = in.readArray(N) ;
        for (int i = 0; i <N ; i++) {
            if (a[i] == 1)list2.add(i) ;
            else list1.add(i) ;
        }

         n = list1.size() ;
         m = list2.size() ;
            dp = new long[n][m] ;


            for(int i=0 ; i<n ; i++)
                for(int j=0 ; j<m ; j++)
                    dp[i][j] = -1 ;

     
        System.out.println(solver(0 , 0 ));








            out.flush();
            out.close();

//        }
//        catch (Exception e){
//            return;
//        }

    }

    

    static ArrayList<Integer> list = new ArrayList<>();
    static boolean A[] = new boolean[2 * 90000001];


    

    

    

    


    

    

    

    

    


    


    


    


}
