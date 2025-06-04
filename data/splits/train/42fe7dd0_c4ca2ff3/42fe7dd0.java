import java.util.*;
 public class Solution {
    static Scanner sc=new Scanner(System.in);
    public static void main(String args[]) {
      int t=sc.nextInt();
      outer:while(t-->0){
        int n=sc.nextInt();
        int[][] ct=new int[n][5];
        int[] len=new int[n];
        for (int i=0;i<n;i++) {
            String s=sc.next();
            len[i]=s.length();
            for(char c:s.toCharArray()){
                ct[i][c-'a']++;
            }
        }
        int mx=0;
        for (int i=0;i<5;i++) {
            int[] diff=new int[n];
            for (int j=0;j<n;j++) {
                diff[j]=ct[j][i]-(len[j]-ct[j][i]);
            }
            Arrays.sort(diff);
            int sum=0,inc=0;
            for(int j=n-1;j>=0;j--){
                sum+=diff[j];
                if (sum>0) {
                    inc++;
                }else {
                    break;
                }
            }
            mx=Math.max(mx,inc);
        }
        System.out.println(mx);
        }
        
      }

    }
