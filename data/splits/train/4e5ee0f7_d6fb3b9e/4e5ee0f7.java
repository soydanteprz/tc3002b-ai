import java.io.*;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Collections;
import java.io.InputStreamReader;
import static java.lang.Math.*;
import static java.lang.System.*;

public class Main1 {







    public static void main(String[] args) throws IOException {

//        try {
//             Scanner in = new Scanner(System.in) ;
        FastScanner in = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);



            int t = in.nextInt() ;
            while (t-- > 0){
                int n = in.nextInt() ;
                int dp[][] = new int[n][5] ;
                String tt[] = new String[n] ;

                for (int i = 0; i <n ; i++) {
                    String  s= in.next() ;
                    tt[i] = s ;
                    for (int j = 0; j <s.length() ; j++) {
                        dp[i][s.charAt(j)-'a']++ ;
                    }
                }
                int max = 0 ;


                for (int i = 0; i <5 ; i++) {
                    ArrayList<Integer>list = new ArrayList<>() ;
                    for (int j = 0; j <n ; j++) {
                        list.add(dp[j][i] - (tt[j].length()-dp[j][i]) );
                    }
                    list.sort(Collections.reverseOrder());
                    int ans = 0 ;
                    int sum = 0 ;
                    for (int curr : list){
                        sum+= curr ;
                        if (sum > 0){
                            ans++ ;
                            max = max(max , ans) ;
                        }
                        else break;
                    }

                }

                System.out.println(max);
            }










            out.flush();
            out.close();


//        } catch (Exception e) {
//            return;
//        }

}

    

    

    static void sort(int ar[]) {
        int n = ar.length;
        ArrayList<Integer> a = new ArrayList<>();
        for (int i = 0; i < n; i++)
            a.add(ar[i]);
        Collections.sort(a);
        for (int i = 0; i < n; i++)
            ar[i] = a.get(i);
    }

    

    

    

    


    


}
