import java.util.*;

public class Soltion{
 
     public static void main(String []args){
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        
        while(t-->0){
            int n = sc.nextInt();
            Integer[] arr = new Integer[n];
            for(int i=0;i<n;i++){
                arr[i] = sc.nextInt();
            }
            String s = sc.next();
            
            List<Integer> blue = new ArrayList<>();
            List<Integer> red = new ArrayList<>();
            
            for(int i=0;i<s.length();i++){
                if(s.charAt(i)=='B'){
                    blue.add(arr[i]);
                }
                else{
                    red.add(arr[i]);
                }
            }
            
            Collections.sort(blue);
            Collections.sort(red);
            int p=1,q=n;
            
            boolean flag = true;
            for(int i=red.size()-1;i>=0;i--){
                if(red.get(i)>q){
                    flag = false;
                    break;
                }
                q--;
            }
            
            for(int i=0;i<blue.size();i++){
                if(blue.get(i)<p){
                    flag = false;
                    break;
                }
                p++;
            }
            
            System.out.println(flag? "Yes" : "No");
        }
     }
}
