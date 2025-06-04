import java.io.*;
import java.util.*;
public class A734C {
    public static void main(String[] args) {
        JS scan = new JS();
        int t = scan.nextInt();
        loop:while(t-->0){
            int n = scan.nextInt();
            String[] arr= new String[n];
            Integer[][] counts = new Integer[5][n];
            for(int i = 0;i<5;i++){
                for(int j = 0;j<n;j++){
                    counts[i][j] = 0;
                }
            }
            for(int i =0;i<n;i++){
                arr[i] = scan.next();
                int[] freq =new int[5];
                for(int j = 0;j<arr[i].length();j++){
                    freq[arr[i].charAt(j)-'a']++;
                }
                for(int j = 0;j<5;j++){
                    counts[j][i] = freq[j]-(arr[i].length()-freq[j]);
                }
            }
            int best = 0;
            for(int i = 0;i<5;i++){
                Arrays.sort(counts[i]);
                int curr = 0;
                int extra = 0;
                for(int j = n-1;j>=0;j--){
                    extra+=counts[i][j];
                    if(extra>0)curr++;
                }
                best = Math.max(best,curr);
            }
            System.out.println(best);

        }
    }
    
}
