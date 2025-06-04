import java.util.*;
import java.lang.*;
import java.io.*;

public class InterestingStrory {
    public static void main(String[] args) throws java.lang.Exception {
        // your code goes here
        try {
            // Scanner sc=new Scanner(System.in);
            FastReader sc = new FastReader();
            int t =sc.nextInt();
            while (t-- > 0) {
                int n=sc.nextInt();
                int[][] occ=new int[n][5];
                for(int i=0;i<n;i++){
                    char[] warr=sc.next().toCharArray();
                    for(char ch:warr){
                        occ[i][ch-'a']++;
                    }
                }
                int ans=Integer.MIN_VALUE;;
                for(int c=0;c<5;c++){
                    int[] sums=new int[n];
                    for(int i=0;i<n;i++){
                        int sum=0;
                        for(int j=0;j<5;j++){
                            if(c==j){
                                sum+=occ[i][j];
                            }
                            else{
                                sum-=occ[i][j];
                            }
                        }
                        sums[i]=sum;
                    }
                    Arrays.sort(sums);
                    int currSum=0;
                    int l=n-1;
                    for(l=n-1;l>=0;l--){
                        if(currSum+sums[l]<=0){
                            break;
                        }
                        else{
                            currSum+=sums[l];
                        }
                    }
                    ans=Math.max(ans,n-l-1);
                }
                System.out.println(ans);
            }
        } catch (Exception e) {
            return;
        }

    }
    

    

    

    
    
    
    

    
}
