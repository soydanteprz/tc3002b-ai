import java.io.*;
import java.util.*;

public class a {

    public static void main(String[] args){

        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while(t-- > 0){

            int n = sc.nextInt();
            String a = sc.next();
            String b = sc.next();

            int change = 0;
            int nochange = 0;
            for(int i=0; i<n; i++){

                if(a.charAt(i) == b.charAt(i)){
                    nochange++;
                }
                else{
                    change++;
                }

            }

            if(change%2 == 0 && nochange%2 == 0){
                
                int count1 = 0;
                for(int i=0; i<n; i++){
                    if(a.charAt(i) != b.charAt(i) && a.charAt(i) == '1'){
                        count1++;
                    }
                }

                if(count1 == change/2){
                    System.out.println(change);
                }
                else{
                    System.out.println(-1);
                }

            }
            else if(change%2 == 0 && nochange%2 == 1){

                // both ways possible

                int count1 = 0;
                int ans = Integer.MAX_VALUE;
                for(int i=0; i<n; i++){
                    if(a.charAt(i) == b.charAt(i) && a.charAt(i) == '1'){
                        count1++;
                    }
                }

                if(count1 == nochange/2 + 1){
                    ans = Math.min(ans, nochange);
                }

                count1 = 0;
                for(int i=0; i<n; i++){
                    if(a.charAt(i) != b.charAt(i) && a.charAt(i) == '1'){
                        count1++;
                    }
                }

                if(count1 == change/2){
                    ans = Math.min(ans, change);
                }

                if(ans == Integer.MAX_VALUE){
                    System.out.println(-1);
                }
                else{
                    System.out.println(ans);
                }

            }
            else if(change%2 == 1 && nochange%2 == 0){
                System.out.println(-1);
            }
            else{

                int count1 = 0;
                for(int i=0; i<n; i++){
                    if(a.charAt(i) == b.charAt(i) && a.charAt(i) == '1'){
                        count1++;
                    }
                }

                if(count1 == nochange/2 + 1){
                    System.out.println(nochange);
                }
                else{
                    System.out.println(-1);
                }

            }

        }

    }

}


